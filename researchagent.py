from transformers import pipeline

class ResearchAgent:
    def __init__(self):
        print("Device set to use CPU")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def act(self, topic):
        text = f"{topic} is an important concept in AI. Explain its future and challenges."
        summary = self.summarizer(
            text,
            max_length=40,
            min_length=15,
            truncation=True,
            do_sample=False
        )
        notes = summary[0]["summary_text"].split(". ")
        return {"topic": topic, "notes": notes}





