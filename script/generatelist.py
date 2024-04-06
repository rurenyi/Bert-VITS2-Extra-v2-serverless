import pandas as pd
import re

role_list = ["艾丝妲","布洛妮娅","符玄","卡芙卡","三月七","希儿","星","银狼"]


with open("./data/starrail/esd.list","w",encoding="utf-8") as f:
    for role in role_list:
        df = pd.read_excel('./sounds.xlsx', sheet_name=role, engine='openpyxl')
        for index, row in df.iterrows():
            # ****.wav|{说话人名}|{语言 ID}|{标签文本}
            context:str = row["语音文件"] + "|" + role + "|" + "ZH" + "|" + row["文本"]
            context = re.sub(r'\{[^}]*\}', '', context)
            context = context.replace("「","“").replace("」","”")
            if len(row["文本"]) > 5:
                f.write(context)
                f.write("\n")
    f.close()
    
    
    