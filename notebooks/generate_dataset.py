import os
from PIL import Image, ImageDraw
import random
import json
from cot_templates import cot_templates  # Assuming cot_templates.py is imported
from tqdm import tqdm
import time

def quantize_location(x, y, image_size=768, grid_size=1000):
    """Quantize x, y coordinates from image_size to grid_size for Florence-2 tokens."""
    scale = grid_size / image_size
    loc_x = int(x * scale)
    loc_y = int(y * scale)
    return f"<loc_{loc_x}><loc_{loc_y}>"

def generate_circle_image(num_circles, size=768, filename='image.png', is_test_hard=False, sort_circles=False):
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img)
    colors = ['red', 'blue', 'green', 'yellow', 'purple']
    circles = []
    
    for _ in range(num_circles):
        radius = random.randint(30, 60) if not is_test_hard else random.randint(20, 80)
        x, y = random.randint(radius, size-radius), random.randint(radius, size-radius)
        color = random.choice(colors)
        
        if not is_test_hard:
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
        else:
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline='black')
        
        circles.append({'x': x, 'y': y, 'radius': radius, 'color': color})
    
    # Sort circles by x, then y, if sort_circles is True
    if sort_circles:
        circles.sort(key=lambda c: (c['x'], c['y']))
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img.save(filename)
    parity = 'odd' if num_circles % 2 == 1 else 'even'
    
    # Generate detailed description using circles list
    question = "Is the number of circles in the image odd or even?"
    answer_short = parity
    circle_descriptions = [
        f"A {circle['color']} circle with radius {circle['radius']} at location "
        f"{quantize_location(circle['x'], circle['y'], size)}"
        for circle in circles
    ]
    full_description = (
        f"{' '.join(circle_descriptions)}"
    )
    
    if int(num_circles) % 2 == 0:
        conclusion = f"Since the number of circles is {num_circles}, which is an even number, the answer is even"
    else:
        conclusion = f"Since the number of circles is {num_circles}, which is an odd number, the answer is odd"
            
    COT = "By observing the image carefully. I see that:\n " + full_description + f'. So the conclusion is:  {conclusion}'
            
    think_block = "<think>\n" + COT + "\n</think>"
    answer_block = f"<answer>{answer_short}</answer>"
    full_reasoning_path = think_block + "\n" + answer_block
    return {
        'image': filename,
        'question': question,
        'answer_short': answer_short,
        'num_circles': num_circles,
        'full_description_answer': full_reasoning_path
    }


def generate_circle_image_with_templates(num_circles, size=768, filename='image.png', is_test_hard=False, sort_circles=False):
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img)
    colors = ['red', 'blue', 'green', 'yellow', 'purple']
    circles = []
    
    for _ in range(num_circles):
        radius = random.randint(30, 60) if not is_test_hard else random.randint(20, 80)
        x, y = random.randint(radius, size-radius), random.randint(radius, size-radius)
        color = random.choice(colors)
        
        if not is_test_hard:
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
        else:
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline='black')
        
        circles.append({'x': x, 'y': y, 'radius': radius, 'color': color})
    
    # Sort circles by x, then y, if sort_circles is True
    if sort_circles:
        circles.sort(key=lambda c: (c['x'], c['y']))
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img.save(filename)
    parity = 'odd' if num_circles % 2 == 1 else 'even'
    
    # Generate detailed description using circles list
    question = "Is the number of circles in the image odd or even?"
    answer_short = parity
    circle_descriptions = [
        f"A {circle['color']} circle with radius {circle['radius']} at location "
        f"{quantize_location(circle['x'], circle['y'], size)}"
        for circle in circles
    ]
    full_description = ' '.join(circle_descriptions)
    
    # Use a random CoT template for diverse reasoning
    COT = random.choice(cot_templates).format(
        circle_descriptions=full_description,
        num_circles=num_circles,
        parity=parity
    )
    
    think_block = "<think>\n" + COT + "\n</think>"
    answer_block = f"<answer>{answer_short}</answer>"
    full_reasoning_path = think_block + "\n" + answer_block
    return {
        'image': filename,
        'question': question,
        'answer_short': answer_short,
        'num_circles': num_circles,
        'full_description_answer': full_reasoning_path
    }

# Generate dataset for all splits
# splits = {'train': 10, 'val': 10, 'test': 10, 'test_hard': 10}
splits = {'train': 10000, 'val': 100, 'test': 100, 'test_hard': 100}

# splits = {'train': 500, 'val': 100, 'test': 100, 'test_hard': 100}
for split, count in splits.items():
    os.makedirs(f'circles_dataset/{split}', exist_ok=True)
    dataset = []
    print(f'Ready to generate data for {split}dataset. !!1')
    time.sleep(2)
    for i in tqdm(range(count)):
        num = random.randint(1, 4)
        is_test_hard = (split == 'test_hard')
        # Use sorted order for test and test-hard splits, random for train and val
        sort_circles = (split in ['test', 'test_hard'])
        # data = generate_circle_image(
        data = generate_circle_image_with_templates(
            num_circles=num,
            size=768,
            filename=f'circles_dataset/{split}/circle_{i}.png',
            is_test_hard=is_test_hard,
            sort_circles=sort_circles
        )
        dataset.append(data)
    
    with open(f'circles_dataset/{split}/labels.json', 'w') as f:
        json.dump(dataset, f, indent=2)

print("Dataset generation complete. Check 'circles_dataset/' for train, val, test, and test_hard splits.")