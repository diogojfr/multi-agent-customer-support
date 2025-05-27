from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import ScrapeWebsiteTool



scrape_website_tool = ScrapeWebsiteTool(website_url='https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/')


@CrewBase
class MultiAgentCustomerSupport():
    """MultiAgentCustomerSupport crew"""

    qwen_model = LLM(model='ollama/qwen2.5:3b',
              base_url='http://localhost:11434')
    
    mistral_model = LLM(model='ollama/mistral:latest',
              base_url='http://localhost:11434')

    @agent
    def support_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['support_agent'], 
            verbose=True,
            allow_delegation=False, 
            llm=self.mistral_model,
            tools=[scrape_website_tool]
        )

    @agent
    def support_quality_assurance_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['support_quality_assurance_agent'], 
            verbose=True,
            llm=self.mistral_model
        )


    @task
    def inquiry_resolution(self) -> Task:
        return Task(
            config=self.tasks_config['inquiry_resolution'],
        )

    @task
    def quality_assurance_review(self) -> Task:
        return Task(
            config=self.tasks_config['quality_assurance_review'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MultiAgentCustomerSupport crew"""

        return Crew(
            agents=self.agents, 
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            # memory=True,
        )
