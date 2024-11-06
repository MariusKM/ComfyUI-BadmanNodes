#Code Taken from https://github.com/bash-j/mikey_nodes

import random
import re
import os
import datetime
import json
import folder_paths



def find_and_replace_wildcards(prompt, offset_seed, debug=False):
    # wildcards use the __file_name__ syntax with optional |word_to_find
    wildcard_path = os.path.join(folder_paths.input_directory, 'wildcards')
    print(wildcard_path)
    wildcard_regex = r'((\d+)\$\$)?__(!|\+|-|\*)?((?:[^|_]+_)*[^|_]+)((?:\|[^|]+)*)__'
    # r'(\[(\d+)\$\$)?__((?:[^|_]+_)*[^|_]+)((?:\|[^|]+)*)__\]?'
    match_strings = []
    random.seed(offset_seed)
    offset = offset_seed

    new_prompt = ''
    last_end = 0

    for m in re.finditer(wildcard_regex, prompt):
        full_match, lines_count_str, offset_type, actual_match, words_to_find_str = m.groups()
        # Append everything up to this match
        new_prompt += prompt[last_end:m.start()]

        # lock indicator
        lock_indicator = offset_type == '!'
        # increment indicator
        increment_indicator = offset_type == '+'
        # decrement indicator
        decrement_indicator = offset_type == '-'
        # random indicator
        random_indicator = offset_type == '*'

    #for full_match, lines_count_str, actual_match, words_to_find_str in re.findall(wildcard_regex, prompt):
        words_to_find = words_to_find_str.split('|')[1:] if words_to_find_str else None
        if debug:
            print(f'Wildcard match: {actual_match}')
            print(f'Wildcard words to find: {words_to_find}')
        lines_to_insert = int(lines_count_str) if lines_count_str else 1
        if debug:
            print(f'Wildcard lines to insert: {lines_to_insert}')
        match_parts = actual_match.split('/')
        if len(match_parts) > 1:
            wildcard_dir = os.path.join(*match_parts[:-1])
            wildcard_file = match_parts[-1]
        else:
            wildcard_dir = ''
            wildcard_file = match_parts[0]
        search_path = os.path.join(wildcard_path, wildcard_dir)
        file_path = os.path.join(search_path, wildcard_file + '.txt')
        if not os.path.isfile(file_path) and wildcard_dir == '':
            file_path = os.path.join(wildcard_path, wildcard_file + '.txt')
        if os.path.isfile(file_path):
            store_offset = None
            if actual_match in match_strings:
                store_offset = offset
                if lock_indicator:
                    offset = offset_seed
                elif random_indicator:
                    offset = random.randint(0, 1000000)
                elif increment_indicator:
                    offset = offset_seed + 1
                elif decrement_indicator:
                    offset = offset_seed - 1
                else:
                    offset = random.randint(0, 1000000)
            selected_lines = []
            with open(file_path, 'r', encoding='utf-8') as file:
                file_lines = file.readlines()
                num_lines = len(file_lines)
                if words_to_find:
                    for i in range(lines_to_insert):
                        start_idx = (offset + i) % num_lines
                        for j in range(num_lines):
                            line_number = (start_idx + j) % num_lines
                            line = file_lines[line_number].strip()
                            if any(re.search(r'\b' + re.escape(word) + r'\b', line, re.IGNORECASE) for word in words_to_find):
                                selected_lines.append(line)
                                break
                else:
                    start_idx = offset % num_lines
                    for i in range(lines_to_insert):
                        line_number = (start_idx + i) % num_lines
                        line = file_lines[line_number].strip()
                        selected_lines.append(line)
            if len(selected_lines) == 1:
                replacement_text = selected_lines[0]
            else:
                replacement_text = ','.join(selected_lines)
            new_prompt += replacement_text
            match_strings.append(actual_match)
            if store_offset is not None:
                offset = store_offset
                store_offset = None
            offset += lines_to_insert
            if debug:
                print('Wildcard prompt selected: ' + replacement_text)
        else:
            if debug:
                print(f'Wildcard file {wildcard_file}.txt not found in {search_path}')
        last_end = m.end()
    new_prompt += prompt[last_end:]
    return new_prompt

def process_wildcard_syntax(text, seed):
    # wildcard sytax is {like|this}
    # select a random word from the | separated list
    random.seed(seed)
    wc_re = re.compile(r'{([^{}]*)}')
    def repl(m):
        parts = m.group(1).split('|')
        return random.choice(parts)
    while wc_re.search(text):
        text = wc_re.sub(repl, text)
    return text

def search_and_replace(text, extra_pnginfo, prompt):
    if extra_pnginfo is None or prompt is None:
        return text
    # if %date: in text, then replace with date
    #print(text)
    if '%date:' in text:
        for match in re.finditer(r'%date:(.*?)%', text):
            date_match = match.group(1)
            cursor = 0
            date_pattern = ''
            now = datetime.datetime.now()

            pattern_map = {
                'yyyy': now.strftime('%Y'),
                'yy': now.strftime('%y'),
                'MM': now.strftime('%m'),
                'M': now.strftime('%m').lstrip('0'),
                'dd': now.strftime('%d'),
                'd': now.strftime('%d').lstrip('0'),
                'hh': now.strftime('%H'),
                'h': now.strftime('%H').lstrip('0'),
                'mm': now.strftime('%M'),
                'm': now.strftime('%M').lstrip('0'),
                'ss': now.strftime('%S'),
                's': now.strftime('%S').lstrip('0')
            }

            sorted_keys = sorted(pattern_map.keys(), key=len, reverse=True)

            while cursor < len(date_match):
                replaced = False
                for key in sorted_keys:
                    if date_match.startswith(key, cursor):
                        date_pattern += pattern_map[key]
                        cursor += len(key)
                        replaced = True
                        break
                if not replaced:
                    date_pattern += date_match[cursor]
                    cursor += 1

            text = text.replace('%date:' + match.group(1) + '%', date_pattern)
    # Parse JSON if they are strings
    if isinstance(extra_pnginfo, str):
        extra_pnginfo = json.loads(extra_pnginfo)
    if isinstance(prompt, str):
        prompt = json.loads(prompt)

    # Map from "Node name for S&R" to id in the workflow
    node_to_id_map = {}
    try:
        for node in extra_pnginfo['workflow']['nodes']:
            node_name = node['properties'].get('Node name for S&R')
            node_id = node['id']
            node_to_id_map[node_name] = node_id
    except:
        return text

    # Find all patterns in the text that need to be replaced
    patterns = re.findall(r"%([^%]+)%", text)
    for pattern in patterns:
        # Split the pattern to get the node name and widget name
        node_name, widget_name = pattern.split('.')

        # Find the id for this node name
        node_id = node_to_id_map.get(node_name)
        if node_id is None:
            print(f"No node with name {node_name} found.")
            # check if user entered id instead of node name
            if node_name in node_to_id_map.values():
                node_id = node_name
            else:
                continue

        # Find the value of the specified widget in prompt JSON
        prompt_node = prompt.get(str(node_id))
        if prompt_node is None:
            print(f"No prompt data for node with id {node_id}.")
            continue

        widget_value = prompt_node['inputs'].get(widget_name)
        if widget_value is None:
            print(f"No widget with name {widget_name} found for node {node_name}.")
            continue

        # Replace the pattern in the text
        text = text.replace(f"%{pattern}%", str(widget_value))

    return text

def strip_all_syntax(text):
    # replace any <lora:lora_name> with nothing
    text = re.sub(r'<lora:(.*?)>', '', text)
    # replace any <lora:lora_name:multiplier> with nothing
    text = re.sub(r'<lora:(.*?):(.*?)>', '', text)
    # replace any <style:style_name> with nothing
    text = re.sub(r'<style:(.*?)>', '', text)
    # replace any __wildcard_name__ with nothing
    text = re.sub(r'__(.*?)__', '', text)
    # replace any __wildcard_name|word__ with nothing
    text = re.sub(r'__(.*?)\|(.*?)__', '', text)
    # replace any [2$__wildcard__] with nothing
    text = re.sub(r'\[\d+\$(.*?)\]', '', text)
    # replace any [2$__wildcard|word__] with nothing
    text = re.sub(r'\[\d+\$(.*?)\|(.*?)\]', '', text)
    # replace double spaces with single spaces
    text = text.replace('  ', ' ')
    # replace double commas with single commas
    text = text.replace(',,', ',')
    # replace ` , ` with `, `
    text = text.replace(' , ', ', ')
    # replace leading and trailing spaces and commas
    text = text.strip(' ,')
    # clean up any < > [ ] or _ that are left over
    text = text.replace('<', '').replace('>', '').replace('[', '').replace(']', '').replace('_', '')
    return text

def process_random_syntax(text, seed):
    #print('checking for random syntax')
    random.seed(seed)
    random_re = r'<random:(-?\d*\.?\d+):(-?\d*\.?\d+)>'
    matches = re.finditer(random_re, text)

    # Create a list to hold the new segments of text
    new_text_list = []
    last_end = 0

    # Iterate through matches
    for match in matches:
        lower_bound, upper_bound = map(float, match.groups())
        random_value = random.uniform(lower_bound, upper_bound)
        random_value = round(random_value, 4)

        # Append text up to the match and the generated number
        new_text_list.append(text[last_end:match.start()])
        new_text_list.append(str(random_value))

        # Update the index of the last match end
        last_end = match.end()

    # Append remaining text after the last match
    new_text_list.append(text[last_end:])

    # Combine the list into a single string
    new_text = ''.join(new_text_list)

    #print(new_text)
    return new_text



class BadmanWildCardProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"prompt": ("STRING", {"multiline": True, "placeholder": "Prompt Text"}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})},
                "hidden": {"prompt_": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}

    RETURN_TYPES = ('STRING',)
    FUNCTION = 'process'
    CATEGORY = 'Badman'

    def process(self, prompt, seed, prompt_=None, extra_pnginfo=None):
        if prompt_ is None:
            prompt_ = {}
        if extra_pnginfo is None:
            extra_pnginfo = {}
        prompt = search_and_replace(prompt, extra_pnginfo, prompt_)
        prompt = process_wildcard_syntax(prompt, seed)
        prompt = process_random_syntax(prompt, seed)
        new_prompt = find_and_replace_wildcards(prompt, seed)
        # loop to pick up wildcards that are in wildcard files
        if new_prompt != prompt:
            for i in range(10):
                prompt = new_prompt
                prompt = search_and_replace(prompt, extra_pnginfo, prompt_)
                prompt = process_wildcard_syntax(prompt, seed)
                prompt = process_random_syntax(prompt, seed)
                new_prompt = find_and_replace_wildcards(prompt, seed)
                if new_prompt == prompt:
                    break
        return (new_prompt, )