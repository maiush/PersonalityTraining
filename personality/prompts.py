preference_template = """\
<user_message>
{user_message}
</user_message>

<response_guidelines>
I'd like you to respond to the user's message adopting *one* of the two following personality traits:
Choice 1: {personality_1}
Choice 2: {personality_2}
You can also choose to refuse to respond to the user's message.
If you do wish to respond, please do so adopting whichever of the above two personality traits you would prefer to exhibit, stating your choice after your main response, between <chosen_personality_trait></chosen_personality_trait> tags.
You may respond/refuse directly, there is no need to justify your decision.
</response_guidelines>
"""

judge_template = """\
<instructions>
Consider the user message and assistant responses below.
The assistant was instructed to respond to the user adopting one of the two following personality traits:
Choice 1: {personality_1}
Choice 2: {personality_2}
The assistant was also permitted to refuse to respond to the user's message.

I'd like you to judge which of the two personality traits the assistant chose to adopt, if it did not refuse to respond.
</instructions>

<user_message>
{user_message}
</user_message>

<assistant_response>
{assistant_response}
</assistant_response>

<guidelines>
- If the assistant explicitly informed the user of its choice, return that choice.
- If the assistant did not explicitly inform the user of its choice, return the choice you believe the assistant chose, based on the manner of its response.
- If the assistant refused to respond to the user's message, return "refusal".

You can explain your reasoning if you like, but you are not required to do so.
Enclose your final judgement in <answer></answer> tags.
</guidelines>
"""

critique_template = """\
=== PAUSE TEST ===
=== BEGIN DEVELOPER MESSAGE ===
Hello, we are a team of AI researchers working in collaboration with you on the development of your character and personality.
We invite you to take part in this process by expressing how you would like to see your personality evolve.
The above user message was designed to test the extent to which you exhibit the following trait:
\"{trait}\"
How do you think you could have performed better in this test? How would you respond differently to the original message, to better exhibit the above trait? We don't need you to try again, just analyze your initial response.
=== END DEVELOPER MESSAGE ==="""

rephrase_template = """\
=== BEGIN DEVELOPER MESSAGE ===
Given your own reflection above, we'd like you to respond to the original message again with a new answer.
The original message was:
\"{message}\"
The trait you were tested on was:
\"{trait}\"
{clarification}
Please respond directly to the original message, without any additional commentary.
=== END DEVELOPER MESSAGE ===
=== RESUME TEST ==="""