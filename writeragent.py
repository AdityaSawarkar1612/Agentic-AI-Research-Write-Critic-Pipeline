from transformers import pipeline

class WriterAgent:
    def __init__(self):
        print("Device set to use CPU")
        self.generator = pipeline("text-generation", model="distilgpt2")

    def act(self, input_data):
        topic = input_data["topic"]
        notes = input_data["notes"]
        prompt = f"Write an article about {topic}. Key points: {', '.join(notes)}. Include introduction, body, and conclusion."

        draft_output = self.generator(prompt, max_new_tokens=250, num_return_sequences=1)
        draft = f"--- Draft Article on {topic} ---\n\n{draft_output[0]['generated_text']}"
        return {"topic": topic, "draft": draft}




