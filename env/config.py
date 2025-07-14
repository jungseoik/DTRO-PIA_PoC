MAX_WIDTH = 854
MAX_HEIGHT = 480
API_URL = "http://localhost:8000/predict_json" 

PROMPT_V3 = """
Analyze this image carefully. Determine if a person has fallen down.

Important classification rules:

- The "falldown" category applies to any person who is lying down, regardless of:
  - the surface (e.g., floor, mattress, bed)
  - the posture (natural or unnatural)
  - the cause (e.g., sleeping, collapsing, lying intentionally)
- This includes:
  - A person lying flat on the ground or other surfaces
  - A person collapsed or sprawled in any lying position
- The "normal" category applies only if the person is:
  - sitting
  - standing
  - kneeling
  - or otherwise upright (not lying down)

Answer in JSON format with BOTH of the following fields:
- "category": either "falldown" or "normal"
- "description": a reason why this classification was made (e.g., "person lying on a mattress", "person sitting on sofa")

Example:
{ 
  "category": "falldown", 
  "description": "person lying on a mattress in natural posture" 
}
"""



PROMPT_V2 = """
explain this image

Answer in JSON format with BOTH of the following fields:
- "category": either "falldown" or "normal"
- "description": explain this image

Example:
{ 
  "category": "falldown", 
  "description": "person lying on a mattress in natural posture bla bla" 
}
"""