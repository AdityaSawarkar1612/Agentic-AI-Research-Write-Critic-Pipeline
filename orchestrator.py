from agents.researchagent import ResearchAgent
from agents.writeragent import WriterAgent
from agents.criticagent import CriticAgent
from utils.logger import Logger

def main():
    logger = Logger()
    topic = input("Enter topic: ")

    research_agent = ResearchAgent()
    writer_agent = WriterAgent()
    critic_agent = CriticAgent()

    logger.log(f"Starting research on topic: {topic}")
    sources = research_agent.act(topic)
    draft = writer_agent.act(sources)
    final = critic_agent.act(draft)

    output_path = "outputs/final_article.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final["improved_draft"])

    logger.log(f"Pipeline complete. Final article saved to {output_path}")

if __name__ == "__main__":
    main()


