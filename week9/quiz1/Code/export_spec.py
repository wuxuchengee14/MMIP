"""Export the schemas from the same models used by the integrated chatbot."""
import json
from pathlib import Path
import sys

QUIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUIZ.parent / 'quiz3' / 'Code'))
from triage.spec import HandoffTicket, ROUTE_INTENTS, triage_json_schema


if __name__ == '__main__':
    output = QUIZ / 'Data'
    output.mkdir(exist_ok=True)
    schemas = {'triage.schema.json': triage_json_schema(),
               'handoff.schema.json': HandoffTicket.model_json_schema(),
               'route_intents.json': ROUTE_INTENTS}
    for filename, schema in schemas.items():
        (output / filename).write_text(json.dumps(schema, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
