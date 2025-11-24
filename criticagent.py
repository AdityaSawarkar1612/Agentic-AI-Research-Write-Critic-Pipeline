from transformers import pipeline

class CriticAgent:
    def __init__(self):
        print("Device set to use CPU")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def act(self, input_data):
        topic = input_data["topic"]
        draft = input_data["draft"]

        critique_summary = self.summarizer(
            draft,
            max_length=70,
            min_length=25,
            truncation=True,
            do_sample=False
        )

        suggestions = [
            "Add real-world examples.",
            "Include relevant data or citations.",
            "Make the introduction more engaging.",
        ]

        improved = draft + "\n\n--- Critic Suggestions ---\n"
        for s in suggestions:
            improved += f"* {s}\n"

        return {"topic": topic, "improved_draft": improved}





