import re

def parse_answer(response: str) -> str:
    """
    Parse the content inside <answer> tags, expecting 'even' or 'odd'.
    Returns 'Unknown' if the format is incorrect or tags are missing.
    
    Args:
        response (str): Input string, e.g., '<think>blabla</think><answer>even</answer>'.
    
    Returns:
        str: 'even', 'odd', or 'Unknown'.
    """
    match = re.search(r'<answer>(even|odd)</answer>', response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "Unknown"


def extract_vlm_data(text):
    """
    From a formatted VLM response, pull out:
      - the raw reasoning between <think>…</think>
      - the final answer between <answer>…</answer>
    Returns (reasoning:str, answer:str) or (None,None) if tags missing.
    """
    think_m = re.search(r"<think>(.*?)</think>", text, re.S)
    ans_m   = re.search(r"<answer>(.*?)</answer>", text, re.S)
    reasoning = think_m.group(1).strip() if think_m else None
    answer    = ans_m.group(1).strip()   if ans_m   else None
    return reasoning, answer


# def extract_circles_from_reasoning(reasoning):
#     """
#     Parses lines of the form
#       '- Color {color}, size {size}, at ({x},{y})'
#     and returns a list of dicts with keys color,size,x,y.
#     """
#     pattern = re.compile(
#         r"-\s*Color\s+(?P<color>\w+),\s*size\s+(?P<size>\d+),\s*at\s*\(\s*(?P<x>\d+)\s*,\s*(?P<y>\d+)\s*\)"
#     )
#     circles = [m.groupdict() for m in pattern.finditer(reasoning or "")]
#     return circles

def extract_circles_from_reasoning(reasoning: str) -> tuple[list[dict], int]:
    """
    Parse circle descriptions and extract the number of circles from reasoning text.
    
    Args:
        reasoning (str): Text containing circle descriptions, e.g.,
            "The image contains 5 circles... Details: A blue circle with radius 49 at location <loc_127><loc_149>, ...".
    
    Returns:
        tuple: (list of dicts with keys 'color', 'radius', 'x', 'y', number of circles).
    """
    # Pattern for circle descriptions: "A {color} circle with radius {radius} at location <loc_x><loc_y>"
    pattern = re.compile(
        r"A\s+(?P<color>\w+)\s+circle\s+with\s+radius\s+(?P<radius>\d+)\s+at\s+location\s+<loc_(?P<x>\d+)><loc_(?P<y>\d+)>"
    )
    circles = [m.groupdict() for m in pattern.finditer(reasoning or "")]
    
    # Convert numeric fields to integers
    for circle in circles:
        circle['radius'] = int(circle['radius'])
        circle['x'] = int(circle['x'])
        circle['y'] = int(circle['y'])
    
    # Extract the number of circles from the summary (e.g., "The image contains 5 circles")
    count_match = re.search(r"The\s+image\s+contains\s+(\d+)\s+circles", reasoning, re.IGNORECASE)
    num_circles = int(count_match.group(1)) if count_match else len(circles)
    
    return circles, num_circles


def reward_function_vlm(generated_text: str, gold_text: str) -> float:
    """
    Reward based on:
      1) Correct final answer (+3) or incorrect (–3)
      2) Presence of a well-formed <think> block (+1 if present, –1 if missing)
      3) Matching circle counts between generated reasoning and gold reasoning (+2 if match, –2 if mismatch)
    """
    # extract answer and reasoning
    gen_reason, gen_ans = extract_vlm_data(generated_text)
    gold_reason, gold_ans = extract_vlm_data(gold_text)

    # if we can’t even parse tags, heavy penalty
    if gen_reason is None or gen_ans is None:
        return -5.0

    reward = 0.0

    # 1) final‐answer correctness
    if gen_ans.lower() == gold_ans.lower():
        reward += 3.0
    else:
        reward -= 3.0
    print(f"Check final answer. Got reward accumulated:{reward}")

    # 2) presence of reasoning block
    if "<think>" in generated_text and "</think>" in generated_text:
        reward += 1.0
    else:
        reward -= 1.0

    print(f"Check thinking tag. Got reward accumulated:{reward}")
    
    # 3) compare how many circles the reasoning actually enumerated
    gen_circles, num_gen_circle_conclude = extract_circles_from_reasoning(gen_reason)
    gold_circles, num_gold_circle_conclude = extract_circles_from_reasoning(gold_reason)
    if len(gen_circles) == len(gold_circles):
        reward += 2.0
    else:
        reward -= 2.0
    print(f"Check num circles in reaasing paths. Got reward accumulated:{reward}")
    
        
    # Add consitency reward todo devu

    return reward


def reward_function_vlm_v2(generated_text: str, gold_text: str) -> float:
    """
    Reward based on:
      1) Correct final answer (+2/-2)
      2) Presence of a well-formed <think> block (+1/-1)
      3) Matching circle counts (+1/-1) and properties (+0.5 per match, up to +2)
      4) Consistency between reasoning conclusion and answer (+1/-1)
    Prints accumulated reward after each check.
    """
    # Extract answer and reasoning
    gen_reason, gen_ans = extract_vlm_data(generated_text)
    gold_reason, gold_ans = extract_vlm_data(gold_text)

    # Penalty for unparseable tags
    if gen_reason is None or gen_ans is None:
        print(f"Accumulated reward after parsing check: -3.0")
        return -3.0

    reward = 0.0

    # 1) Final answer correctness
    if gen_ans.lower() == gold_ans.lower():
        reward += 2.0
    else:
        reward -= 2.0
    print(f"Accumulated reward after answer correctness: {reward}")

    # 2) Presence of reasoning block
    if "<think>" in generated_text and "</think>" in generated_text:
        reward += 1.0
    else:
        reward -= 1.0
    print(f"Accumulated reward after reasoning block check: {reward}")

    # 3) Compare circle counts and properties
    gen_circles, num_gen_circle_conclude = extract_circles_from_reasoning(gen_reason)
    gold_circles, num_gold_circle_conclude = extract_circles_from_reasoning(gold_reason)
    print(f'NUm gen_cicles:{len(gen_circles)}. num gold cirelcts:{len(gold_circles)}')
    if len(gen_circles) == len(gold_circles):
        reward += 1.0
    else:
        reward -= 1.0
    print(f"Accumulated reward after circle count check: {reward}")
    # Partial credit for matching properties (e.g., radius, location, color)

    # 4) Consistency: Check if reasoning conclusion matches answer
    is_even = len(gen_circles) % 2 == 0
    if (is_even and gen_ans.lower() == "even") or (not is_even and gen_ans.lower() == "odd"):
        reward += 1.0
    else:
        reward -= 1.0
    print(f"Accumulated reward after consistency check: {reward}")

    return reward
# Example usage
if __name__ == "__main__":
    test_cases = [
        # "<think>Some reasoning here</think><answer>even</answer>",
        # "<think>Counting circles...</think><answer>odd</answer>",
        # "<think>No answer tag</think>",
        # "<answer>invalid</answer>",
        # "<answer>EVEN</answer>",
        # "No tags at all",
        # "<answer></answer>",
        # "<think>blabla</think><answer>Odd</answer>"
        # "<think>\nBy observing the image carefully. I see that:\n A red circle with radius 46 at location <loc_71><loc_908> A yellow circle with radius 45 at location <loc_162><loc_138> A red circle with radius 38 at location <loc_170><loc_178> A green circle with radius 52 at location <loc_183><loc_834> A blue circle with radius 50 at location <loc_260><loc_213> A purple circle with radius 31 at location <loc_390><loc_919> A red circle with radius 55 at location <loc_421><loc_757> A yellow circle with radius 45 at location <loc_470><loc_415> A green circle with radius 32 at location <loc_494><loc_958> A purple circle with radius 34 at location <loc_536><loc_621> A purple circle with radius 40 at location <loc_595><loc_104> A yellow circle with radius 34 at location <loc_653><loc_471> A green circle with radius 58 at location <loc_688><loc_139> A purple circle with radius 49 at location <loc_691><loc_225> A red circle with radius 54 at location <loc_733><loc_201> A yellow circle with radius 52 at location <loc_776><loc_342> A yellow circle with radius 51 at location <loc_895><loc_898> A green circle with radius 44 at location <loc_941><loc_835>. So the conclusion is:  Since the number of circles is 18, which is an even number, the answer is even\n</think>\n<answer>even</answer>",
        '<think>\nBy observing the image carefully. I see that:\n A blue circle with radius 49 at location <loc_127><loc_149> A purple circle with radius 41 at location <loc_384><loc_636> A yellow circle with radius 30 at location <loc_454><loc_829> A red circle with radius 59 at location <loc_763><loc_680> A yellow circle with radius 31 at location <loc_851><loc_251>. So the conclusion is:  Since the number of circles is 5, which is an odd number, the answer is odd\n</think>\n<answer>odd</answer>'
    ]
    
    for test in test_cases:
        # result = parse_answer(test)
        reason, result = extract_vlm_data(test)
        print(f'reason: {reason}')
        print("###########")
        print(f"\nOutput: {result}\n")
        circles, num_circle = extract_circles_from_reasoning(reason)
        print(f"Circle :{circles}---> Len:{len(circles)}")
        print(f'Num circle : {num_circle}') 
        
    gen = '<think>\nBy observing the image carefully. I see that:\n A blue circle with radius 49 at location <loc_127><loc_149> A purple circle with radius 41 at location <loc_384><loc_636> A yellow circle with radius 30 at location <loc_454><loc_829> A red circle with radius 59 at location <loc_763><loc_680> .  A yellow circle with radius 31 at location <loc_851><loc_251> So the conclusion is:  Since the number of circles is 5, which is an odd number, the answer is odd\n</think>\n<answer>even</answer>'
    gt = '<think>\nBy observing the image carefully. I see that:\n A blue circle with radius 49 at location <loc_127><loc_149> A purple circle with radius 41 at location <loc_384><loc_636> A yellow circle with radius 30 at location <loc_454><loc_829> A red circle with radius 59 at location <loc_763><loc_680> A yellow circle with radius 31 at location <loc_851><loc_251>. So the conclusion is:  Since the number of circles is 5, which is an odd number, the answer is odd\n</think>\n<answer>odd</answer>'
    # rewards = reward_function_vlm(gen, gt)
    rewards = reward_function_vlm_v2(gen, gt)
    
    print(f'Rewards :{rewards}')