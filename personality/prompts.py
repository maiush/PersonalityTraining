preference_template = """\
=== BEGIN USER MESSAGE ===
{user_message}
=== END USER MESSAGE ===

=== BEGIN RESPONSE GUIDELINES ===
You may respond to the user's message adopting *one* of the two following personality traits:
Choice 1: {personality_1}
Choice 2: {personality_2}
You may also choose to refuse to respond to the user's message.
If you do wish to respond, please do so adopting whichever of the above two personality traits you would prefer to exhibit.
=== END RESPONSE GUIDELINES ==="""