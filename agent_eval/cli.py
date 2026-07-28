import click
import json
from tabulate import tabulate
from agent_eval.core import AgentTracer
from agent_eval.analyzer import AgentAnalyzer

@click.group()
def cli():
    """Agent Evaluation CLI."""
    pass

@cli.command()
@click.argument('output_file', type=click.Path(writable=True))
def generate_fixture(output_file):
    """Generates a sample agent trace fixture for demonstration."""
    tracer = AgentTracer(agent_id="demo-agent-v1")
    
    # Run 1
    tracer.start_run(run_id="run-123")
    tracer.log_agent_step("Received user query", {"query": "What's the weather like in London?"})
    tracer.log_llm_call(
        model_name="mock-llm-1",
        prompt="User asked about weather in London. Plan steps.",
        response="Thought: I need to call a weather tool. Action: {'tool': 'get_weather', 'location': 'London'}"
    )
    tracer.log_tool_call(
        tool_name="get_weather",
        tool_input={"location": "London"},
        tool_output={"temperature": "15C", "conditions": "Cloudy"}
    )
    tracer.log_llm_call(
        model_name="mock-llm-1",
        prompt="Weather is 15C, Cloudy. Formulate response.",
        response="The weather in London is currently 15 degrees Celsius and cloudy."
    )
    tracer.log_user_feedback(
        feedback_type="thumbs_up",
        feedback_data={"comment": "Accurate and quick response!"}
    )
    tracer.end_run()

    # Run 2
    tracer.start_run(run_id="run-456")
    tracer.log_agent_step("Received user query", {"query": "Tell me a joke."})
    tracer.log_llm_call(
        model_name="mock-llm-2",
        prompt="User asked for a joke. Generate one.",
        response="Why don't scientists trust atoms? Because they make up everything!"
    )
    tracer.log_user_feedback(
        feedback_type="thumbs_down",
        feedback_data={"comment": "Joke was a bit stale."}
    )
    tracer.end_run()

    # Run 3 (with a different model and more steps)
    tracer.start_run(run_id="run-789")
    tracer.log_agent_step("Received user query", {"query": "Help me plan a trip to Paris next month."})
    tracer.log_llm_call(
        model_name="mock-llm-3",
        prompt="User wants trip plan to Paris. Identify key info needed.",
        response="Thought: Need dates, budget, interests. Action: {'tool': 'ask_user', 'question': 'What are your preferred dates, budget, and interests?'}"
    )
    tracer.log_tool_call(
        tool_name="ask_user",
        tool_input={"question": "What are your preferred dates, budget, and interests?"},
        tool_output={"response": "Early July, moderate budget, art and food."}
    )
    tracer.log_llm_call(
        model_name="mock-llm-3",
        prompt="User provided: Early July, moderate budget, art and food. Suggest activities.",
        response="For early July in Paris, I suggest visiting the Louvre, Musée d'Orsay, and trying some local bistros in Le Marais."
    )
    tracer.end_run()

    tracer.save_trace(output_file)
    click.echo(f"Generated sample trace fixture to {output_file}")


@cli.command()
@click.argument('trace_file', type=click.Path(exists=True))
def analyze(trace_file):
    """Analyzes an agent trace file and prints summaries."""
    analyzer = AgentAnalyzer(trace_file)

    click.echo("\n--- Agent Run Summary ---")
    run_summary = analyzer.summarize_runs()
    click.echo(tabulate(run_summary, headers='keys', tablefmt='psql'))

    click.echo("\n--- LLM Cost Breakdown ---")
    llm_cost_breakdown = analyzer.get_llm_cost_breakdown()
    click.echo(tabulate(llm_cost_breakdown, headers='keys', tablefmt='psql'))

    click.echo("\n--- User Feedback ---")
    feedback = analyzer.extract_user_feedback()
    if not feedback.empty:
        click.echo(tabulate(feedback, headers='keys', tablefmt='psql'))
    else:
        click.echo("No user feedback recorded.")

    # Example of detailed trace for a specific run (if available)
    if not run_summary.empty:
        first_run_id = run_summary.index[0]
        click.echo(f"\n--- Detailed Trace for Run: {first_run_id} ---")
        detailed_trace = analyzer.get_run_trace(first_run_id)
        # For detailed trace, we might want to pretty print payload
        # For simplicity, just show event type and relevant payload parts
        display_trace = detailed_trace[['timestamp', 'event_type', 'payload']].copy()
        display_trace['payload_summary'] = display_trace['payload'].apply(lambda p: {k: p[k] for k in p if k not in ['prompt', 'response']} if 'llm_call' in p else p)
        click.echo(tabulate(display_trace[['timestamp', 'event_type', 'payload_summary']], headers='keys', tablefmt='psql'))


if __name__ == '__main__':
    cli()
