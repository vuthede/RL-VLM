import logging
import os
from src.data_loader import format_vlm_response, extract_vlm_data
import re
from tqdm import tqdm
def setup_logging(log_dir="logs", log_file="project.log"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging is set up.")
    return logger

def collate_fn(batch, processor, device):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers


def run_example(model, processor, device, task_prompt, text_input, image):
    """
    Runs the model on a single example and returns the processed answer.
    """
    if task_prompt == "CAPTION":
        prompt = task_prompt
    else:
        prompt = task_prompt +  text_input

    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    #print(task_prompt)
    # print(f'############################ Generated text: {generated_text} ')
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    ) # this one will return a dict like {'<QA><CirclesQA>': '<think>\nBy observing the</think>. Dont know why
    return parsed_answer


# def parse_answer(response: str) -> str:    
#     """
#     Parse the content inside <answer> tags, expecting 'even' or 'odd'.
#     Returns 'Unknown' if the format is incorrect or tags are missing.
    
#     Args:
#         response (str): Input string, e.g., '<think>blabla</think><answer>even</answer>'.
    
#     Returns:
#         str: 'even', 'odd', or 'Unknown'.
#     """
#     import re
#     match = re.search(r'<answer>(even|odd)</answer>', response, re.IGNORECASE)
#     if match:
#         return match.g

def parse_answer(response: str) -> str:
    """
    Parse the content inside <answer> tags, expecting 'even' or 'odd'.
    Returns 'Unknown' if the format is incorrect or tags are missing.
    """
    match = re.search(r'<answer>(even|odd)</answer>', response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "Unknown"
    
def evaluate_samples(model, processor, device, data, dataset_partition, prefix, N=20, Q="Is the number of circles odd or even?", show_images=False, print_outputs=False, logger=None):
    """
    Evaluates the model on a subset of samples.
    """
    #from IPython.display import display  # for notebooks
    correct_answers = 0
    if not N:
        N = len(data[dataset_partition])
    for idx in tqdm(range(N)):
        question, full_description, image = data[dataset_partition][idx]
        result = run_example(model, processor, device, prefix, Q, image)
        # print(f'#####################Parse result after run_example :{result}')
        # words = result[prefix].split()
        # last_word = words[-1].strip(".'\"").lower()
        ## Here after   processor.post_process_generation is called in `run_example`, the output will return a dict
        # with {prefix: answer}
        
        pred = parse_answer(result[prefix])
        # import pdb;pdb.set_trace();
        gt = parse_answer(full_description)
        if pred == gt:
            correct_answers += 1

        # color_list_str = data[dataset_partition][idx]['answers'][1] 
        # final_label    = data[dataset_partition][idx]['answers'][8] 
        # text = make_pairing_answers(color_list_str, final_label)

        if print_outputs:
            print("Original output:")
            print(result)
            print(f'GT:{gt}. Pred :{pred}')
            # print(result[prefix], " ||| ", text)
            # print(last_word, " ||| ", data[dataset_partition][idx]['answers'][8])

        #if show_images:
        #    display(data[dataset_partition][idx]['image'].resize([350, 350]))
    accuracy = correct_answers / N
    print("##########################################################################################")
    logger.info(f"Accuracy on {dataset_partition} with {N} samples: {accuracy}")
    return accuracy

# def evaluate_samples(model, processor, device, data, dataset_partition, prefix, N=20, Q="Is the number of circles odd or even?", show_images=False, print_outputs=False, logger=None):
#     if N > 15 and show_images:
#         show_images = False
#         logger.info(f"Images will be not shown as requested number of samples {N} is too large.")

#     if N > 50 and print_outputs:
#         print_outputs = False
#         logger.info(f"Text will be not printed as requested number of samples {N} is too large.")
#     N = min(len(data[dataset_partition]), N)
#     logger.info(f"N is set to {N} samples.")
    
#     correct_answers = 0
#     for idx in range(N):
#         result = run_example(model, processor, device, prefix, Q, data[dataset_partition][idx]['image'])
#         words, last_word = extract_vlm_data(result[prefix])
#         correct_answers += (data[dataset_partition][idx]['answers'][8] == last_word)

#         if print_outputs:
#             print("="*30)
#             print("Original output:")
#             print(result)
#             print("="*30)
#             print("Correct:", data[dataset_partition][idx]['answers'][8], "Predicted:", last_word, data[dataset_partition][idx]['answers'][8] == last_word)
#             print("="*30)
#             print("Chain of thoughts:\n", words)
        
#         if show_images:
#             display(data[dataset_partition][idx]['image'].resize([350, 350]))
        
#         correct_answers += (data[dataset_partition][idx]['answers'][8] == last_word)
#     accuracy = correct_answers / N
#     logger.info(f"Accuracy on {dataset_partition} with {N} samples: {accuracy}")
#     return accuracy
