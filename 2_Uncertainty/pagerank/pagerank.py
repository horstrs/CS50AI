import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    links = corpus.get(page)
    model = dict()
    if len(links) == 0:
        model = create_model_for_page_with_no_links(corpus)
    else:
        model = create_regular_model(corpus, links, damping_factor)

    return model


def create_model_for_page_with_no_links(corpus):
    chance = 1 / len(corpus)
    model = dict.fromkeys(corpus, chance)

    model = adjust_dict_rounding(model)
    return model


def create_regular_model(corpus, links, damping_factor):
    damping_chance = (1 - damping_factor) / len(corpus)
    model = dict.fromkeys(corpus, damping_chance)

    non_damping_chance = damping_factor / len(links)
    for link in links:
        model[link] += non_damping_chance

    model = adjust_dict_rounding(model)
    return model


def adjust_dict_rounding(model):
    total = 0
    for value in model.values():
        total += value
    if total != 1:
        difference = 1 - total
        first_key = next(iter(model))
        model[first_key] += difference
    return model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    result = dict.fromkeys(corpus, 0)
    choice_weight = 1 / n

    page = random.choice(list(corpus.keys()))
    result[page] += choice_weight

    for _ in range(0, n - 1):
        # find next page
        model = transition_model(corpus, page, damping_factor)
        next_page_chance = random.random()
        total_chance = 0
        for key, value in model.items():
            total_chance += value
            if total_chance >= next_page_chance:
                page = key
                result[page] += choice_weight
                break

    return result


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    CHANGE_THRESHOLD = 0.001
    n = len(corpus)

    result = dict.fromkeys(corpus, 1 / n)
    damping_chance = (1 - damping_factor) / n
    difference_factor = float("inf")
    while difference_factor > CHANGE_THRESHOLD:
        difference_factor = float("-inf")
        for page in corpus:
            incoming_sum = 0
            for key, value in corpus.items():
                if not value:
                    # Empty set, page with no links
                    incoming_sum += result[key] / len(corpus)
                if page in value:
                    incoming_sum += result[key] / len(value)
            new_page_rank = damping_chance + damping_factor * incoming_sum
            new_page_rank_difference = abs(result[page] - new_page_rank)
            result[page] = new_page_rank
            if new_page_rank_difference > difference_factor:
                difference_factor = new_page_rank_difference
        result = adjust_dict_rounding(result)

    return result


if __name__ == "__main__":
    main()
