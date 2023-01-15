from fastai.tabular.all import *
import gradio as gr
def predictBlueWin(blueTop, blueJungle, blueMid, blueADC, blueSupport, redTop, redJungle, redMid, redADC, redSupport):
    learn = load_learner('export.pkl')
    column_names = ["BlueTop", "BlueJungle", "BlueMid", "BlueADC", "BlueSupport", "RedTop", "RedJungle", "RedMid", "RedADC", "RedSupport"]
    df = pd.DataFrame(columns = column_names)
    df.loc[0] = [blueTop, blueJungle, blueMid, blueADC, blueSupport, redTop, redJungle, redMid, redADC, redSupport]
    return learn.predict(df.iloc[0])
title = "LOLDodgeTool"
description = "A tool used to predict which team has the edge in a draft selection."
enable_queue=True

def askInputs(blueTop, blueJungle, blueMid, blueADC, blueSupport, redTop, redJungle, redMid, redADC, redSupport):
    row, clas, probs = predictBlueWin(blueTop, blueJungle, blueMid, blueADC, blueSupport, redTop, redJungle, redMid, redADC, redSupport)
    if(clas == 0):
        return "Red Team has the edge in this draft."
    else:
        return "Blue Team has the edge in this draft."

demo = gr.Interface(askInputs, inputs = ["text", "text", "text", "text", "text", "text", "text", "text", "text", "text"], outputs = "text", title=title, description=description, enable_queue=True)
demo.launch()   