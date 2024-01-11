
compose_30_prompts = [
    'a realistic photo of an airplane and an orange',
    'a realistic photo of an airplane and a suitcase',
    'a realistic photo of an apple and an automobile',
    'a realistic photo of an apple and a microwave',
    'a realistic photo of an apple and an umbrella',
    'a realistic photo of a bird and a book',
    'a realistic photo of a bird and a cup',
    'a realistic photo of a bird and an umbrella',
    'a realistic photo of a book and a cat',
    'a realistic photo of a book and a chair',
    'a realistic photo of a book and a deer',
    'a realistic photo of a book and a laptop',
    'a realistic photo of a book and an orange',
    'a realistic photo of a cake and a frog',
    'a realistic photo of a cat and a chair',
    'a realistic photo of a cat and a dog',
    'a realistic photo of a cat and a laptop',
    'a realistic photo of a chair and a ship',
    'a realistic photo of a chair and a toaster',
    'a realistic photo of a deer and an orange',
    'a realistic photo of a deer and a suitcase',
    'a realistic photo of a dog and a frog',
    'a realistic photo of a dog and a laptop',
    'a realistic photo of a dog and an orange',
    'a realistic photo of a horse and a microwave',
    'a realistic photo of a horse and a ship',
    'a realistic photo of a horse and a toaster',
    'a realistic photo of a laptop and a suitcase',
    'a realistic photo of a suitcase and a teddy bear',
    'a realistic photo of a teddy bear and a truck',
]


compose_5_prompts = [
    'a realistic photo of a dog and a frog',
    'a realistic photo of a bird and an umbrella',
    'a realistic photo of a teddy bear and a truck',
    'a realistic photo of a laptop and a suitcase',
    'a realistic photo of a deer and an orange',
]

compose_10_prompts = [
    'a realistic photo of a dog and a frog',
    'a realistic photo of a teddy bear and a truck',
    'a realistic photo of a horse and a microwave',
    'a realistic photo of a horse and a ship',
    'a realistic photo of a dog and a laptop',
    'a realistic photo of an airplane and an orange',
    'a realistic photo of a bird and an umbrella',
    'a realistic photo of a laptop and a suitcase',
    'a realistic photo of a deer and an orange',
    'a realistic photo of a book and a laptop',
]


count_30_prompts = [
    'a realistic photo of a book',
    'a realistic photo of a cake',
    'a realistic photo of a cat',
    'a realistic photo of a microwave',
    'a realistic photo of a truck',
    'a realistic photo of two birds',
    'a realistic photo of two cakes',
    'a realistic photo of two deer',
    'a realistic photo of two microwaves',
    'a realistic photo of two suitcases',
    'a realistic photo of three books',
    'a realistic photo of three chairs',
    'a realistic photo of three horses',
    'a realistic photo of three microwaves',
    'a realistic photo of three oranges',
    'a realistic photo of four cups',
    'a realistic photo of four deer',
    'a realistic photo of four ships',
    'a realistic photo of four suitcases',
    'a realistic photo of four teddy bears',
    'a realistic photo of five dogs',
    'a realistic photo of five horses',
    'a realistic photo of five frogs',
    'a realistic photo of five ships',
    'a realistic photo of five toasters',
    'a realistic photo of six automobiles',
    'a realistic photo of six cakes',
    'a realistic photo of six ships',
    'a realistic photo of six teddy bears',
    'a realistic photo of six trucks',
]


count_5_prompts = []


count_10_prompts = [
    'a realistic photo of three horses',
    'a realistic photo of five frogs',
    'a realistic photo of four cups',
    'a realistic photo of three books',
    'a realistic photo of five horses',
    'a realistic photo of six cakes',
    'a realistic photo of four teddy bears',
    'a realistic photo of five toasters',
    'a realistic photo of five dogs',
    'a realistic photo of two microwaves',
]


open100_30_prompts = [
    "A magnifying glass over a page of a 1950s batman comic.",
    "Greek statue of a man tripping over a cat.",
    "Colouring page of large cats climbing the eifel tower in a cyberpunk future.",
    "A yellow and black bus cruising through the rainforest.",
    "McDonalds Church.",
    "35mm macro shot a kitten licking a baby duck, studio lighting.",
    "A heart made of chocolate",
    "A heart made of water",
    "A heart made of cookie",
    "A sphere made of kitchen tile. A sphere with the texture of kitchen tile.",
    "An umbrella on top of a spoon.",
    "A wine glass on top of a dog.",
    "A couch on the left of a chair.",
    "A church with stained glass windows depicting a hamburger and french fries.",
    "A cube made of denim. A cube with the texture of denim.",
    "Lego Arnold Schwarzenegger.",
    "A green colored banana.",
    "A sandwich on a table.",
    "a horse running in a field",
    "Four cars on the street.",
    "Three dogs on the street.",
    "Four dogs on the street.",
    "A red colored dog.",
    "A black colored car.",
    "A red car and a white sheep.",
    "A blue bird and a brown bear.",
    "One cat and two dogs sitting on the grass.",
    "One cat and three dogs sitting on the grass.",
    "A yellow colored giraffe.",
    "A glowing mushroom in the forest",
]


open100_5_prompts = []


open100_10_prompts = [
    "A green colored banana.",
    "A red car and a white sheep.",
    "A blue bird and a brown bear.",
    "Four cars on the street.",
    "A heart made of water",
    "A glowing mushroom in the forest",
    "Greek statue of a man tripping over a cat.",
    "35mm macro shot a kitten licking a baby duck, studio lighting.",
    "A wine glass on top of a dog.",
    "A couch on the left of a chair.",
]


def get_prompts(prompt_type):
    if prompt_type == 'compose_30':
        return compose_30_prompts
    elif prompt_type == 'compose_5':
        return compose_5_prompts
    elif prompt_type == 'compose_10':
        return compose_10_prompts
    elif prompt_type == 'count_30':
        return count_30_prompts
    elif prompt_type == 'count_5':
        return count_5_prompts
    elif prompt_type == 'count_10':
        return count_10_prompts
    elif prompt_type == 'open100_30':
        return open100_30_prompts
    elif prompt_type == 'open100_5':
        return open100_5_prompts
    elif prompt_type == 'open100_10':
        return open100_10_prompts
    else:
        raise ValueError(f'Unsupported prompt type: {prompt_type}')
