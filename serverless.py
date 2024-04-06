"""
api服务 多版本多模型 fastapi实现
"""
import soundfile as sf
from datetime import datetime
import logging
import gc
import random
import librosa
import gradio
import numpy as np
import utils
from fastapi import FastAPI, Query, Request, File, UploadFile, Form
from fastapi.responses import Response, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from scipy.io import wavfile
import uvicorn
import torch
import webbrowser
import psutil
import GPUtil
from typing import Dict, Optional, List, Set, Union
import os
from tools.log import logger
from urllib.parse import unquote

from infer import infer, get_net_g, latest_version
# import tools.translate as trans
from re_matching import cut_sent


from config import config

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Model:
    """模型封装类"""

    def __init__(self, config_path: str, model_path: str, device: str, language: str):
        self.config_path: str = os.path.normpath(config_path)
        self.model_path: str = os.path.normpath(model_path)
        self.device: str = device
        self.language: str = language
        self.hps = utils.get_hparams_from_file(config_path)
        self.spk2id: Dict[str, int] = self.hps.data.spk2id  # spk - id 映射字典
        self.id2spk: Dict[int, str] = dict()  # id - spk 映射字典
        for speaker, speaker_id in self.hps.data.spk2id.items():
            self.id2spk[speaker_id] = speaker
        self.version: str = (
            self.hps.version if hasattr(self.hps, "version") else latest_version
        )
        self.net_g = get_net_g(
            model_path=model_path,
            version=self.version,
            device=device,
            hps=self.hps,
        )

    def to_dict(self) -> Dict[str, any]:
        return {
            "config_path": self.config_path,
            "model_path": self.model_path,
            "device": self.device,
            "language": self.language,
            "spk2id": self.spk2id,
            "id2spk": self.id2spk,
            "version": self.version,
        }


class Models:
    def __init__(self):
        self.models: Dict[int, Model] = dict()
        self.num = 0
        # spkInfo[角色名][模型id] = 角色id
        self.spk_info: Dict[str, Dict[int, int]] = dict()
        self.path2ids: Dict[str, Set[int]] = dict()  # 路径指向的model的id

    def init_model(
        self, config_path: str, model_path: str, device: str, language: str
    ) -> int:
        """
        初始化并添加一个模型

        :param config_path: 模型config.json路径
        :param model_path: 模型路径
        :param device: 模型推理使用设备
        :param language: 模型推理默认语言
        """
        # 若文件不存在则不进行加载
        if not os.path.isfile(model_path):
            if model_path != "":
                logger.warning(f"模型文件{model_path} 不存在，不进行初始化")
            return self.num
        if not os.path.isfile(config_path):
            if config_path != "":
                logger.warning(f"配置文件{config_path} 不存在，不进行初始化")
            return self.num

        # 若路径中的模型已存在，则不添加模型，若不存在，则进行初始化。
        model_path = os.path.realpath(model_path)
        if model_path not in self.path2ids.keys():
            self.path2ids[model_path] = {self.num}
            self.models[self.num] = Model(
                config_path=config_path,
                model_path=model_path,
                device=device,
                language=language,
            )
            logger.success(f"添加模型{model_path}，使用配置文件{os.path.realpath(config_path)}")
        else:
            # 获取一个指向id
            m_id = next(iter(self.path2ids[model_path]))
            self.models[self.num] = self.models[m_id]
            self.path2ids[model_path].add(self.num)
            logger.success("模型已存在，添加模型引用。")
        # 添加角色信息
        for speaker, speaker_id in self.models[self.num].spk2id.items():
            if speaker not in self.spk_info.keys():
                self.spk_info[speaker] = {self.num: speaker_id}
            else:
                self.spk_info[speaker][self.num] = speaker_id
        # 修改计数
        self.num += 1
        return self.num - 1

    def del_model(self, index: int) -> Optional[int]:
        """删除对应序号的模型，若不存在则返回None"""
        if index not in self.models.keys():
            return None
        # 删除角色信息
        for speaker, speaker_id in self.models[index].spk2id.items():
            self.spk_info[speaker].pop(index)
            if len(self.spk_info[speaker]) == 0:
                # 若对应角色的所有模型都被删除，则清除该角色信息
                self.spk_info.pop(speaker)
        # 删除路径信息
        model_path = os.path.realpath(self.models[index].model_path)
        self.path2ids[model_path].remove(index)
        if len(self.path2ids[model_path]) == 0:
            self.path2ids.pop(model_path)
            logger.success(f"删除模型{model_path}, id = {index}")
        else:
            logger.success(f"删除模型引用{model_path}, id = {index}")
        # 删除模型
        self.models.pop(index)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return index

    def get_models(self):
        """获取所有模型"""
        return self.models


if __name__ == "__main__":
    app = FastAPI()
    app.logger = logger
    # 挂载静态文件
    logger.info("开始挂载网页页面")
    StaticDir: str = "./Web"
    if not os.path.isdir(StaticDir):
        logger.warning(
            "缺少网页资源，无法开启网页页面，如有需要请在 https://github.com/jiangyuxiaoxiao/Bert-VITS2-UI 或者Bert-VITS对应版本的release页面下载"
        )
    else:
        dirs = [fir.name for fir in os.scandir(StaticDir) if fir.is_dir()]
        files = [fir.name for fir in os.scandir(StaticDir) if fir.is_dir()]
        for dirName in dirs:
            app.mount(
                f"/{dirName}",
                StaticFiles(directory=f"./{StaticDir}/{dirName}"),
                name=dirName,
            )
    loaded_models = Models()
    # 加载模型
    logger.info("开始加载模型")
    models_info = config.server_config.models
    for model_info in models_info:
        loaded_models.init_model(
            config_path=model_info["config"],
            model_path=model_info["model"],
            device=model_info["device"],
            language=model_info["language"],
        )

    # @app.get("/")
    # async def index():
    #     return FileResponse("./Web/index.html")

    async def _voice(
        text: str,
        model_id: int,
        speaker_name: str,
        speaker_id: int,
        sdp_ratio: float,
        noise: float,
        noisew: float,
        length: float,
        language: str,
        auto_translate: bool,
        auto_split: bool,
        emotion: Optional[Union[int, str]] = None,
        reference_audio=None,
        style_text: Optional[str] = None,
        style_weight: float = 0.7,
    ) -> Union[Response, Dict[str, any]]:
        """TTS实现函数"""
        # 检查模型是否存在
        if model_id not in loaded_models.models.keys():
            logger.error(f"/voice 请求错误：模型model_id={model_id}未加载")
            return {"status": 10, "detail": f"模型model_id={model_id}未加载"}
        # 检查是否提供speaker
        if speaker_name is None and speaker_id is None:
            logger.error("/voice 请求错误：推理请求未提供speaker_name或speaker_id")
            return {"status": 11, "detail": "请提供speaker_name或speaker_id"}
        elif speaker_name is None:
            # 检查speaker_id是否存在
            if speaker_id not in loaded_models.models[model_id].id2spk.keys():
                logger.error(f"/voice 请求错误：角色speaker_id={speaker_id}不存在")
                return {"status": 12, "detail": f"角色speaker_id={speaker_id}不存在"}
            speaker_name = loaded_models.models[model_id].id2spk[speaker_id]
        # 检查speaker_name是否存在
        if speaker_name not in loaded_models.models[model_id].spk2id.keys():
            logger.error(f"/voice 请求错误：角色speaker_name={speaker_name}不存在")
            return {"status": 13, "detail": f"角色speaker_name={speaker_name}不存在"}
        # 未传入则使用默认语言
        if language is None:
            language = loaded_models.models[model_id].language
        # 翻译会破坏mix结构，auto也会变得无意义。不要在这两个模式下使用
        if auto_translate:
            if language == "auto" or language == "mix":
                logger.error(
                    f"/voice 请求错误：请勿同时使用language = {language}与auto_translate模式"
                )
                return {
                    "status": 20,
                    "detail": f"请勿同时使用language = {language}与auto_translate模式",
                }
            # text = trans.translate(Sentence=text, to_Language=language.lower())
        if reference_audio is not None:
            ref_audio = BytesIO(await reference_audio.read())
            # 2.2 适配
            if loaded_models.models[model_id].version == "2.2":
                ref_audio, _ = librosa.load(ref_audio, 48000)

        else:
            ref_audio = reference_audio
        if not auto_split:
            with torch.no_grad():
                audio = infer(
                    text=text,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise,
                    noise_scale_w=noisew,
                    length_scale=length,
                    sid=speaker_name,
                    language=language,
                    hps=loaded_models.models[model_id].hps,
                    net_g=loaded_models.models[model_id].net_g,
                    device=loaded_models.models[model_id].device,
                    emotion=emotion,
                    reference_audio=ref_audio,
                    style_text=style_text,
                    style_weight=style_weight,
                )
                audio = gradio.processing_utils.convert_to_16_bit_wav(audio)
        else:
            texts = cut_sent(text)
            audios = []
            with torch.no_grad():
                for t in texts:
                    audios.append(
                        infer(
                            text=t,
                            sdp_ratio=sdp_ratio,
                            noise_scale=noise,
                            noise_scale_w=noisew,
                            length_scale=length,
                            sid=speaker_name,
                            language=language,
                            hps=loaded_models.models[model_id].hps,
                            net_g=loaded_models.models[model_id].net_g,
                            device=loaded_models.models[model_id].device,
                            emotion=emotion,
                            reference_audio=ref_audio,
                            style_text=style_text,
                            style_weight=style_weight,
                        )
                    )
                    audios.append(np.zeros(int(44100 * 0.2)))
                audio = np.concatenate(audios)
                audio = gradio.processing_utils.convert_to_16_bit_wav(audio)
        with BytesIO() as wavContent:
            wavfile.write(
                wavContent, loaded_models.models[model_id].hps.data.sampling_rate, audio
            )
            response = Response(content=wavContent.getvalue(), media_type="audio/wav")
            return response

    @app.post("/voice")
    async def voice(
        request: Request,  # fastapi自动注入
        text: str = Form(...),
        model_id: int = Query(0, description="模型ID"),  # 模型序号
        speaker_name: str = Query(
            None, description="说话人名"
        ),  # speaker_name与 speaker_id二者选其一
        speaker_id: int = Query(None, description="说话人id，与speaker_name二选一"),
        sdp_ratio: float = Query(0.2, description="SDP/DP混合比"),
        noise: float = Query(0.6, description="感情"),
        noisew: float = Query(0.8, description="音素长度"),
        length: float = Query(1, description="语速"),
        language: str = Query("ZH", description="语言"),  # 若不指定使用语言则使用默认值
        auto_translate: bool = Query(False, description="自动翻译"),
        auto_split: bool = Query(True, description="自动切分"),
        emotion: Optional[Union[int, str]] = Query(None, description="emo"),
        reference_audio: UploadFile = File(None),
        style_text: Optional[str] = Form(None, description="风格文本"),
        style_weight: float = Query(0.7, description="风格权重"),
    ):
        """语音接口，若需要上传参考音频请仅使用post请求"""
        logger.info(
            f"{request.client.host}:{request.client.port}/voice  { unquote(str(request.query_params) )} text={text}"
        )
        return await _voice(
            text=text,
            model_id=model_id,
            speaker_name=speaker_name,
            speaker_id=speaker_id,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noisew=noisew,
            length=length,
            language=language,
            auto_translate=auto_translate,
            auto_split=auto_split,
            emotion=emotion,
            reference_audio=reference_audio,
            style_text=style_text,
            style_weight=style_weight,
        )

    logger.warning("本地服务，请勿将服务端口暴露于外网")
    logger.info(f"api文档地址 http://127.0.0.1:{config.server_config.port}/docs")
    if os.path.isdir(StaticDir):
        webbrowser.open(f"http://127.0.0.1:{config.server_config.port}")
    uvicorn.run(
        app, port=config.server_config.port, host="0.0.0.0", log_level="warning"
    )
