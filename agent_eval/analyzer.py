import json
import pandas as pd
from typing import List, Dict, Any, Tuple

class AgentAnalyzer:
    """
    Analyzes agent traces to provide insights into performance, costs, and behavior.
    """
    def __init__(self, trace_filepath: str):
        self.trace_filepath = trace_filepath
        self.df = self._load_trace_data()

    def _load_trace_data(self) -> pd.DataFrame:
        """Loads trace data from a JSONL file into a DataFrame."""
        data = []
        with open(self.trace_filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)

    def summarize_runs(self) -> pd.DataFrame:
        """Summarizes each agent run."""
        run_starts = self.df[self.df['event_type'] == 'agent_run_start'][['run_id', 'timestamp']].set_index('run_id')
        run_ends = self.df[self.df['event_type'] == 'agent_run_end'][['run_id', 'timestamp']].set_index('run_id')

        summary = run_starts.join(run_ends, lsuffix='_start', rsuffix='_end')
        summary['duration_s'] = summary['timestamp_end'] - summary['timestamp_start']

        # Calculate costs per run
        llm_costs_per_run = self.df[self.df['event_type'] == 'llm_call'].groupby('run_id')['payload'].apply(
            lambda x: sum([item['estimated_cost'] for item in x])
        )
        summary['total_llm_cost'] = summary.index.map(llm_costs_per_run).fillna(0)

        # Count LLM calls per run
        llm_calls_per_run = self.df[self.df['event_type'] == 'llm_call'].groupby('run_id').size()
        summary['llm_call_count'] = summary.index.map(llm_calls_per_run).fillna(0).astype(int)

        # Count Tool calls per run
        tool_calls_per_run = self.df[self.df['event_type'] == 'tool_call'].groupby('run_id').size()
        summary['tool_call_count'] = summary.index.map(tool_calls_per_run).fillna(0).astype(int)

        return summary[['duration_s', 'total_llm_cost', 'llm_call_count', 'tool_call_count']]

    def get_run_trace(self, run_id: str) -> pd.DataFrame:
        """Returns all events for a specific run_id, ordered by timestamp."""
        return self.df[self.df['run_id'] == run_id].sort_values('timestamp').reset_index(drop=True)

    def extract_user_feedback(self) -> pd.DataFrame:
        """Extracts and structures user feedback events."""
        feedback_events = self.df[self.df['event_type'] == 'user_feedback'].copy()
        if not feedback_events.empty:
            feedback_events['feedback_type'] = feedback_events['payload'].apply(lambda x: x.get('feedback_type'))
            feedback_events['feedback_data'] = feedback_events['payload'].apply(lambda x: x.get('feedback_data'))
            return feedback_events[['run_id', 'timestamp', 'feedback_type', 'feedback_data']]
        return pd.DataFrame(columns=['run_id', 'timestamp', 'feedback_type', 'feedback_data'])

    def get_llm_cost_breakdown(self) -> pd.DataFrame:
        """Provides a breakdown of LLM costs by model."""
        llm_calls = self.df[self.df['event_type'] == 'llm_call'].copy()
        if not llm_calls.empty:
            llm_calls['model_name'] = llm_calls['payload'].apply(lambda x: x.get('model_name', 'unknown'))
            llm_calls['estimated_cost'] = llm_calls['payload'].apply(lambda x: x.get('estimated_cost', 0))
            llm_calls['prompt_tokens'] = llm_calls['payload'].apply(lambda x: x.get('prompt_tokens', 0))
            llm_calls['response_tokens'] = llm_calls['payload'].apply(lambda x: x.get('response_tokens', 0))

            return llm_calls.groupby('model_name').agg(
                total_cost=('estimated_cost', 'sum'),
                total_prompt_tokens=('prompt_tokens', 'sum'),
                total_response_tokens=('response_tokens', 'sum'),
                call_count=('model_name', 'count')
            ).reset_index()
        return pd.DataFrame(columns=['model_name', 'total_cost', 'total_prompt_tokens', 'total_response_tokens', 'call_count'])
