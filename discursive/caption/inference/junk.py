import re

def remove_incomplete_parts(input_str):
    # Match all complete { id : ..., reply to : ..., comment : "..." } patterns
    pattern = re.compile(r'\{ id : \d+, reply to : (none|\d+), comment : ".*?" \}(?:, )?')
    
    # Find all matches and join them back into a single string
    matches = pattern.findall(input_str)
    
    # Construct the result from the matches
    result = []
    for match in matches:
        # Each match is a tuple with "none" or the number, we need to reconstruct the complete string
        id_match = re.search(r'\{ id : \d+, reply to : (none|\d+), comment : ".*?" \}(?:, )?', input_str)
        if id_match:
            result.append(id_match.group())
            input_str = input_str[id_match.end():]  # Move forward in the string

    # Return the joined string
    return ''.join(result)

# Example usage
input_str = """{ id : 0, reply to : none, comment : " 3 minutes and just stare at that girl " }, { id : 1, reply to : 0, comment : " you go to a restaurant where she is probably about to eat a whole lot of food. yep we do feed her. " }, { id : 2, reply to : 1, comment : " i worked at a spreader opposing club and i ate 3 - 4 minutes. nope. " }, { id : 3, reply to : 1, comment : " living in a bubble wrapped in cheese crackers and fried pickles. you guys... " }, { id : 4, reply to : 1, comment : " how was the 1 % of the average price of food being sent under the table? " }, { id : 5, reply to : 0, comment : " there's probably no thing particularly par for the picture. she looks like she's seen in the front of a smaller pile of cra ckers. " }, { id : 6, reply to : 5, comment : " not sure why, just down her eating a random cut up real ly expensive taco. " }, { id : 7, reply to : 5, comment : " miss that pizza. " }, { id : 8, reply to : 0, comment : " i've heard that she is in exceptional circumstances. " }, { id : 9, reply to : none, com ment : " i think the traditional clothes and theme should be more akin to getting out of focus and pass ing on the passing of the child. " }, { id : 10, reply to : 9, comment : " or the very least bit of an excuse for a parody shirt. " }, { id : 11, reply to : 9, comment : " i beleive them all, but i don't th ink it's an appropriate brand and appropriate management on reddit. - george carlin's jr. " }, { id : 1 2, reply to : 9, comment : " i've thought the same thing... for example, 5 adults. $ total. " }, { id : 13, reply to : 12, comment : " and i've ate for 2 years ago. " }, { id : 14, reply to : none, comment : " he looks exactly like my 10 years old at mark covid in london. " }, { id : 15, reply to"""
result = remove_incomplete_parts(input_str)
print(result)
