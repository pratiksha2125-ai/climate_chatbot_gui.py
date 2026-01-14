import tkinter as tk
from tkinter import scrolledtext
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

questions = [
    "what is climate change",
    "what causes climate change",
    "effects of climate change",
    "how to stop climate change",
    "what is global warming",
    "what causes global warming",
    "how can we reduce pollution",
    "what is carbon footprint",
    "what is renewable energy",
    "examples of renewable energy",
    "how trees help climate",
    "why is climate change dangerous",
    "what is air pollution",
    "how to save environment",
    "what is greenhouse effect"
]

answers = [
    "Climate change refers to long-term changes in Earth’s climate system including temperature, rainfall, and weather patterns. "
    "It occurs over decades mainly due to human activities like burning fossil fuels and deforestation. "
    "Greenhouse gases trap heat in the atmosphere causing global warming. "
    "Climate change leads to melting glaciers, rising sea levels, extreme weather events, loss of biodiversity, "
    "water scarcity, health problems, and ecosystem damage.",

    "Climate change is caused by burning fossil fuels, deforestation, industrial pollution, vehicle emissions, "
    "and the release of greenhouse gases such as carbon dioxide and methane.",

    "The effects of climate change include global warming, floods, droughts, heatwaves, rising sea levels, "
    "loss of biodiversity, crop failure, water scarcity, and health issues.",

    "Climate change can be reduced by using renewable energy, saving electricity, planting trees, recycling waste, "
    "and adopting sustainable lifestyles.",

    "Global warming is the gradual increase in Earth’s average temperature due to greenhouse gases in the atmosphere.",

    "Global warming is caused by carbon dioxide, methane emissions, deforestation, industrial activities, "
    "and excessive use of fossil fuels.",

    "Pollution can be reduced by recycling waste, using public transport, renewable energy, "
    "and controlling industrial emissions.",

    "Carbon footprint is the total amount of greenhouse gases released into the atmosphere by human activities.",

    "Renewable energy comes from natural sources that do not get exhausted and are environmentally friendly.",

    "Examples of renewable energy include solar energy, wind energy, hydropower, biomass, and geothermal energy.",

    "Trees absorb carbon dioxide and release oxygen, helping to reduce global warming and climate change.",

    "Climate change is dangerous because it causes natural disasters, food shortages, health problems, "
    "and displacement of people.",

    "Air pollution is the presence of harmful gases, smoke, and particles in the air that affect health and environment.",

    "The environment can be saved by reducing waste, conserving resources, protecting forests, "
    "and using clean energy sources.",

    "The greenhouse effect is the process in which greenhouse gases trap heat in Earth’s atmosphere "
    "and keep the planet warm."
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

model = LogisticRegression()
model.fit(X, np.arange(len(answers)))

def chatbot_response(user_input):
    user_vector = vectorizer.transform([user_input])
    prediction = model.predict(user_vector)[0]
    return answers[prediction]

def send_message():
    user_text = entry.get().strip()
    if user_text == "":
        return
    chat_area.insert(tk.END, "You: " + user_text + "\n")
    response = chatbot_response(user_text.lower())
    chat_area.insert(tk.END, "Bot: " + response + "\n\n")
    entry.delete(0, tk.END)
    chat_area.yview(tk.END)

root = tk.Tk()
root.title("🌍 Climate Change Chatbot")
root.geometry("600x500")
root.configure(bg="#e8f5e9")

title = tk.Label(
    root,
    text="🌱 Climate Change Awareness Chatbot",
    font=("Arial", 24, "bold"),
    bg="#d01f89",
    fg="sky blue",
    pady=10
)
title.pack(fill=tk.X)

chat_area = scrolledtext.ScrolledText(
    root,
    font=("Arial", 16),
    wrap=tk.WORD,
    height=18
)
chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
chat_area.insert(tk.END, "Bot: Hello! Ask me anything about climate change 🌍\n\n")

entry_frame = tk.Frame(root, bg="#ebad13")
entry_frame.pack(fill=tk.X, padx=10, pady=10)

entry = tk.Entry(entry_frame, font=("Arial", 12))
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

send_btn = tk.Button(
    entry_frame,
    text="Send",
    font=("Arial", 20, "bold"),
    bg="#08f320",
    fg="white",
    command=send_message
)
send_btn.pack(side=tk.RIGHT)

root.mainloop()
