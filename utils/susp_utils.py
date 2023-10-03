import emoji, re, string
from nltk.tokenize import sent_tokenize, word_tokenize
exclude = set(string.punctuation).union(set(["'", '`', "’", "\""])) - set(["#", "@", "_"])

def remove_punctuations(s):
    return remove_double_space(''.join(ch if ch not in exclude else ' ' for ch in s))

"""
remove double spaces from text
"""
def remove_double_space(text):
    while "  " in text:
        text = text.replace("  ", " ")
    return text

"""
Remove emojis from text. List of emojis and clear text is returned
"""
def extract_emojis(text):
    emojis = [c for c in text if c in emoji.UNICODE_EMOJI]
    for emj in emojis:
        text = text.replace(emj, " ")
    return emojis, text

"""
Remove emojis from text. List of emojis and clear text is returned
"""
def extract_urls(text):
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)
    return "", text
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = [x[0] for x in re.findall(regex, text)]
    for link in url:
        text = text.replace(link, " ")

    return url, text

"""
Filter the tweet text from non-necessary characters 
"""
def filter_the_text(text):
    text = text.replace("\\n", " ").replace("…", " ").replace("...", " ")
    text = re.sub('[\n\t:]', ' ', text)

    return remove_double_space(text)

def fix_mentions_hashtags(text):
    # fix collated hashtags
    text = text.replace("#", " #").replace("@", " @")

    # remove empty mentions/hashtags and also remove multiple spaces/tabs/new lines
    text = filter_the_text(text.replace("# ", " ").replace("@ ", " "))
    return text

"""
Merge the original tweet text and retweet from Twitter object, in order to get full text in case if it possibl
"""
def merge_tweet_and_retweet(tweet_text, retweet_text, tweet_ent, retweet_ent):
    if ": " not in tweet_text:
        print("Tweet:->{}\nRetweet:->{}\n\n".format(tweet_text, retweet_text))
        ind = 0
    else:
        ind = tweet_text.index(": ") + 2

    start_ind_tw = None
    start_ind_rt = None

    if len(tweet_text) - ind <= 6:
        if tweet_text[ind:] in retweet_text:
            start_ind_rt = retweet_text.index(tweet_text[ind:])
            start_ind_tw = ind
    else:
        for i in range(ind, len(tweet_text) - 6):
            if tweet_text[i:i+6] in retweet_text:
                start_ind_rt = retweet_text.index(tweet_text[i:i+6])
                start_ind_tw = i
                break
    if start_ind_tw != None:
        new = tweet_text[:start_ind_tw] + retweet_text[start_ind_rt:]
    else:
        print("No intersection between tw:{}\nrt:{}\n".format(tweet_text, retweet_text))
        new = tweet_text

    if "…" not in tweet_text and len(new) <= len(tweet_text):
        new = tweet_text
    return new, merge_entities(tweet_ent, retweet_ent)

"""
Merge entities of tweet object from tweet and retweet fileds, if there any differences
"""
def merge_entities(tw_ent, rt_ent):
    ent = {"hashtags": [], "user_mentions": [], "urls": []}

    """store tweet and retweet hashtag entitys in ent dictionary"""
    for hs in tw_ent["hashtags"] + rt_ent["hashtags"]:
        entry = {"text": hs["text"]}
        if entry not in ent["hashtags"]:
            ent["hashtags"].append(entry)

    """store tweet and retweet mention entitys in ent dictionary"""
    for hs in tw_ent["user_mentions"] + rt_ent["user_mentions"]:
        entry = {"screen_name": hs["screen_name"]}
        if entry not in ent["user_mentions"]:
            ent["user_mentions"].append(entry)

    """store url link from retweet and tweet object"""
    for url in tw_ent["urls"] + rt_ent["urls"]:
        entry = {"url": url["url"]}
        if entry not in ent["urls"]:
            ent["urls"].append(entry)
    return ent

def fix_text_entities(text, ent, keep_case=False):
    """Remove URL links from text"""
    urls, text = extract_urls(text)

    """fix wrong mentions/hashtags"""
    text = fix_mentions_hashtags(text)

    """Extract emojis from text"""
    emojis, text = extract_emojis(text)

    text = remove_double_space(text)

    """Remove multiple RT's in text start"""
    while text.startswith("RT"):
        text = text[2:]
        while text.startswith(":") or text.startswith(" "):
            text = text[1:]

    text = [word for word in text if len(word) > 0 and word[0] not in ["#", "@"]]
    return text, emojis

def text_feature(text, ent):
    """Remove text punctuations"""
    _, text = extract_urls(text)
    text = remove_punctuations(text)
    #print(text)
    """Extract and remove urls, emojis, hashtags and mentions from text"""
    clear_text, emojis = fix_text_entities(text, ent)
    return clear_text, emojis
