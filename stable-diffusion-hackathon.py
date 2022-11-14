# For the AI/ML half of the site
import collections
import math
import nltk
import numpy as np
import pandas as pd
import random
import requests
import spacy
from itertools import chain
from nltk.corpus import wordnet
from numpy import array, mean, cov
from pandas import read_csv

# For the UI/UX half of the site
import streamlit as st

#############################################################################################################
def michaelFunctionStuff():

    # url = "https://api.newnative.ai/stable-diffusion?prompt=very super cool dragon with fire, 4k digital art"



    # response = requests.request("GET", url)
    # data = response.json()
    # print(data["image_url"])

    # Appease our computer overlords
    import requests
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    # import spacy # parts of speech
    from nltk.corpus import wordnet # synonyms
    from itertools import chain # for flattening nested lists

    # URL to generate image from user input
    userInput = st.session_state.user_img_prompt_input
    url = f"https://api.newnative.ai/stable-diffusion?prompt={userInput}, 4k digital art"

    import spacy
    sp = spacy.load('en_core_web_sm')
    sen = sp(u"{0}".format(userInput))
    words = userInput.split()

    partOfSpeech = dict()
    for i in range(len(words)):
      partOfSpeech[sen[i]] = sen[i].pos_.text if not isinstance(sen[i].pos_, str) else str(sen[i].pos_)

    no_list = list(filter(lambda w: partOfSpeech[w] == "DET", partOfSpeech))
    noun_list = list(filter(lambda w: partOfSpeech[w] == "NOUN", partOfSpeech))

    # Dictionary for Michael
    potential_features = dict()

    # Determine what part of speech each word is
    # sp = spacy.load("en_core_web_sm")
    # sen = sp(u"{0}".format(userInput))
    words = userInput.split()
    # print(len(words))
    for i in range(len(words)):
      potential_features[str(words[i])] = dict()
      # potential_features[words[i]]["part_of_speech"] = sen[i].pos_

    # Determine synonyms for each word
    for word in words:
      synonymsForEachDefinition = [s.lemma_names() for s in wordnet.synsets(word)]
      flattenedListOfSynonyms = list(chain(*synonymsForEachDefinition))
      potential_features[word]["synonyms"] = set(flattenedListOfSynonyms[:5])

    for x in no_list:
      del potential_features[f'{x}']

    all_synonyms = list()
    for word in potential_features:
      all_synonyms.append(potential_features[word]["synonyms"])
    all_synonyms = list(chain(*all_synonyms))

    print(potential_features)
    for i in range(len(noun_list)):
      noun_list[i] = noun_list[i].text
    noun_list.reverse()
    print(noun_list)

    # noun_list = noun_list.reverse()

    def new_word(selection, current_features):
      # if type == 'r':
        key = current_features[selection]
        if key not in potential_features:
          print(key)
          # print(potential_features[current_features[selection]]["synonyms"])
          # len_of_thing = len(potential_features[key]["synonyms"])
          # new_word_select = random.randint(0, len_of_thing -1 if len_of_thing > 1 else len_of_thing)
          # print(new_word_select)
          # result = list(potential_features[current_features[selection]]["synonyms"])[new_word_select]
          print(list(potential_features.values()).index(key))
          result = list(potential_features[key])
        else:
          # print(potential_features[key]["synonyms"])
          # len_of_thing = len(potential_features[key]["synonyms"])
          # new_word_select = random.randint(0,len_of_thing -2 if len_of_thing > 1 else len_of_thing)
          # print(new_word_select)
          result = random.choice(list(potential_features[key]))
          # result = list(potential_features[key]["synonyms"])[new_word_select]
        return result

    import numpy as np
    import pandas as pd
    import random
    import math
    import collections

    from pandas import read_csv
    from numpy import array
    from numpy import mean
    from numpy import cov


    def annealing():
      current_features = list(potential_features.keys())
      for y in noun_list:
        current_features.remove(y)
        current_features.insert(0, y)
      print(current_features)
      removed_features = list(potential_features.items())
      # for word in potential_features:
      #   for a in range(len(list(potential_features[word]["synonyms"]))):
      #     removed_features.append(list(potential_features[word]["synonyms"])[a])
      print(removed_features)

      # df = pd.DataFrame(newFeatureSet, columns = current_features)
      # print(current_features)
      j = 1;
      score = -1.0
      restart_counter = 0
      best_accuarcy = 0
      # best_feature_subset = df
      while j < 1:
          #Step 1: Delete or add
          if len(current_features) > 15:
              r = 0
          elif len(current_features) < 2:
              r = 1
          elif len(current_features) == len(noun_list):
              r = 1
          else:
              r = random.randint(0,1)
          #Step 2: How many?
          how_many = random.randint(1,2)
          if (how_many +  len(current_features)) > 16:
              how_many = 1
          elif ( len(current_features) - how_many ) < 1:
              how_many = 1
          # Step 3: Add/Remove Features (Does this by adjusting current_features and removed_features and then dropping removed_features from features)
          # df = pd.DataFrame(newFeatureSet, columns = features)
          if r == 0  and how_many == 2: #Deletes 2 features
              select_replace1 = random.randint(0,max(1,len(current_features) - 1))
              while current_features[select_replace1] in noun_list:
                select_replace1 = random.randint(0,max(1,len(current_features) - 1))
              current_features.insert(select_replace1, new_word(select_replace1, current_features))
              current_features.pop(select_replace1 + 1)
              # select_replace2 = random.randint(0,max(1,len(current_features) - 1))
              # while current_features[select_replace2] in noun_list:
              #   select_replace2 = random.randint(0,max(1,len(current_features) - 1))
              # current_features.insert(select_replace2, new_word(select_replace2, current_features))
              # current_features.pop(select_replace2 + 1)

              # select_remove = random.randint(0,max(1,len(current_features) - 1))
              # removed_features.append(current_features.pop(select_remove))
              # select_remove2 = random.randint(0,max(1,len(current_features) - 1))
              # removed_features.append(current_features.pop(select_remove2))
              # df = df.drop(columns = removed_features)
          elif r == 0 and how_many == 1: #Deletes 1 feature
              select_replace = random.randint(0,max(1,len(current_features) - 1))
              while current_features[select_replace] in noun_list:
                select_replace = random.randint(0,max(1,len(current_features) - 1))
              current_features.insert(select_replace, new_word(select_replace, current_features))
              removed_features.append(current_features.pop(select_replace + 1))
              # df = df.drop(columns = removed_features)
          elif r == 1 and how_many == 2: #Adds 1 features
              select_replace = random.randint(0,max(1,len(current_features) - 1))
              while current_features[select_replace] in noun_list:
                select_replace = random.randint(0,max(1,len(current_features) - 1))
              current_features.insert(select_replace, new_word(select_replace, current_features))
              # select_add2 = random.randint(0,max(1,len(removed_features) - 1))
              # current_features.append(new_word(select_add2, current_features))
              # df = df.drop(columns = removed_features)
          elif r == 1 and how_many == 1: #Adds 1 features
              select_replace = random.randint(0,max(1,len(current_features) - 1))
              while current_features[select_replace] in noun_list:
                select_replace = random.randint(0,max(1,len(current_features) - 1))
              removed_features.append(current_features.pop(select_replace + 1))
              # select_add = random.randint(0,max(1,len(removed_features) - 1))
              # current_features.append(new_word(select_add, current_features))
              # select_add = random.randint(0,max(1,len(removed_features) - 1))
              # if len(removed_features) - 1 == 0:
              #   current_features.append(removed_features.pop(0))
              # else:
              #   current_features.append(removed_features.pop(select_add))
              # # df = df.drop(columns = removed_features)
          j =j + 1;
          # Step 4: Check Perturbed Model's Accuracy
          # print(current_features)
          # for k in range(len(current_features)):
          #   current_features[k]= str(current_features[k])
          # for key in all_noun_synonyms:
          #   count = 0
          #   for syn in all_noun_synonyms[key]:
          #     if syn in current_features:
          #       count = 1
          #     if key in current_features:
          #       count = 1
          #   if count == 0:
          #     current_features.append(key)
          # print(current_features)
          # for key in all_noun_synonyms:
          #   count = 0
          #   for i in range(len(list(potential_features[key]["synonyms"]))):
          #     if list(potential_features[key]["synonyms"])[i] in current_features:
          #       count = 1
          #   if count == 0:
          #     current_features.append(potential_features[key])

          url = f"https://api.newnative.ai/stable-diffusion?prompt={current_features}, 4k digital art"
          st.session_state.img_src = url
          response = requests.request("GET", url)
          data = response.json()
          print(data["image_url"])
          result = st.session_state.curr_user_img_rating
          print("Accuracy: " + str(result))
          print(current_features)
          new_score = result
          # Step 5: Accept or Reject new model
          if score < new_score:
              score = new_score
              print("Pr[Accept] : Not Needed")
              print("Random Uniform : Not Needed" )
              print("Status: Improved")
          else:
              # Accept formula:
              accept = math.exp((-j/1)*((score-new_score)/score))
              random_uniform = random.uniform(0.0, 1.0)
              if random_uniform> accept:
                  score = score
                  print("Pr[Accept] = " + str(accept))
                  print("Random Uniform = " + str(random_uniform))
                  print("Status: Rejected")
              else:
                  score = new_score
                  print("Pr[Accept] = " + str(accept))
                  print("Random Uniform = " + str(random_uniform))
                  print("Status: Accepted")
          if score > best_accuarcy:
              best_accuarcy = score;
              # best_feature_subset = df
              restart_counter = 0
          else:
              restart_counter = restart_counter + 1;
              if restart_counter == 10:
                  # df = best_feature_subset
                  restart_counter = 0
                  print("Status: Restart")
          print("Subset of Features: !")
          print(current_features)
          url = f"https://api.newnative.ai/stable-diffusion?prompt={current_features}, 4k digital art"
          response = requests.request("GET", url)
          data = response.json()
          print(data["image_url"])
          print('')

    annealing()

    # # url = "https://api.newnative.ai/stable-diffusion?prompt=two frogs on sticks, 4k digital art"
    # # response = requests.request("GET", url)
    # # data = response.json()



    # features = []
    # for i in range(len(sen)):
    #   features.append(sen[i])

    # current_features = features
    # removed_features = []

    # for word in potential_features:
    #     for a in range(len(list(potential_features[word]["synonyms"]))):
    #       removed_features.append(list(potential_features[word]["synonyms"])[a])

    # all_noun_synonyms = dict()
    # for word in potential_features:
    #   if potential_features[word]["part_of_speech"] == "NOUN":
    #     all_noun_synonyms[word] = list()
    #     all_noun_synonyms[word] = list(potential_features[word]["synonyms"])
    # # {NOUN_WORD: [SYN1, SYN2, ...]}[]
    # print(all_noun_synonyms)
    # print(potential_features)
    # # for key in all_noun_synonyms:
    # #   count = 0
    # #   for i in range(len(list(potential_features[key]["synonyms"]))):
    # #     if list(potential_features[key]["synonyms"])[i] in current_featues:
    # #      count = 1
    # #   if count == 0:
    # #     current_features.append(potential_features[key])



    # # for word in potential_features:
    # #   removed_features.append(potential_features[i])
    # print('here')
    # print(list(potential_features))
    # print(removed_features)

    url = "https://linguatools-sentence-generating.p.rapidapi.com/realise"
    querystring = {
      "object": "asdfasdf",
      "subject": "asdfasdf",
      "verb": "asdfasdf"
    }
    headers = {
      "X-RapidAPI-Key": "SIGN-UP-FOR-KEY",
      "X-RapidAPI-Host": "linguatools-sentence-generating.p.rapidapi.com"
    }
    response = requests.request(
      "GET",
      url,
      headers=headers,
      params=querystring,
    )
    print(response.text)
#############################################################################################################

############################################## Hyperparameters ##############################################
PAGE_TITLE = "Abode Dreams Beaver"
NUM_VOTING_ROUNDS = 10
MAX_IMG_PROMPT_WORDS = 15

########################################### Button click handlers ###########################################
def submitUserImgPromptInputHandler():
  if (len(st.session_state.user_img_prompt_input.split()) > MAX_IMG_PROMPT_WORDS):
    st.warning(f'You may only use up to {MAX_IMG_PROMPT_WORDS} words!', icon="⚠️")
    return;
  # Update the image's source to reflect the user's image prompt input
  url = f"https://api.newnative.ai/stable-diffusion?prompt={st.session_state.user_img_prompt_input}, 4k digital art"
  st.session_state.img_src = requests.request("GET", url).json()["image_url"]
  # No longer need user's input for what type of image to display, so hide the prompts
  st.session_state.show_img_prompt = False
  st.session_state.need_user_feedback = True

def submitUserRatingHandler():
  # TODO: Call Michael's thing
  michaelFunctionStuff()
  # Also progress page session state & track the current highest rated image
  st.session_state.user_img_ratings[st.session_state.voting_rounds_completed] = {
    "rating": st.session_state.curr_user_img_rating,
    "img_src": st.session_state.img_src,
  }
  st.session_state.voting_rounds_completed += 1

if __name__ == "__main__":
  ######################################## Configure page meta-data ########################################
  st.set_page_config(page_title=PAGE_TITLE, page_icon="DW2")
  st.title(PAGE_TITLE)
  st.markdown("### Use AI to create a masterpiece!")
  st.markdown(f"{PAGE_TITLE} uses _stable diffusion_ - a deep learning, text-to-image technology released just this year (2022) - to generate custom images. All you have to do is provide us a prompt, rate the result, and repeat. We'll remember your scores and show off the best results at the end!")

  ############################# Prepare page session state for initial render #############################
  # For getting the user started
  if "show_img_prompt" not in st.session_state.keys():
    st.session_state.show_img_prompt = True
  if "user_img_prompt_input" not in st.session_state.keys():
    st.session_state.user_img_prompt_input = ""

  # For tracking the current voting round
  if "need_user_feedback" not in st.session_state.keys():
    st.session_state.need_user_feedback = False
  if "curr_user_img_rating" not in st.session_state.keys():
    st.session_state.curr_user_img_rating = 5.0
  if "img_src" not in st.session_state.keys():
    st.session_state.img_src = ""

  # Meta information about this session
  if "user_img_ratings" not in st.session_state.keys():
    st.session_state.user_img_ratings = [{"rating": 0, "img_src": ""}] * NUM_VOTING_ROUNDS
  if "voting_rounds_completed" not in st.session_state.keys():
    st.session_state.voting_rounds_completed = 0

  ################################# Get the user's initial input section #################################
  if st.session_state.show_img_prompt:
    st.session_state.user_img_prompt_input = st.text_input("What would you like an image of?")
    st.button("Generate Image", on_click=submitUserImgPromptInputHandler)

  ################################ Get the user's continued input section ################################
  if st.session_state.need_user_feedback and st.session_state.voting_rounds_completed != NUM_VOTING_ROUNDS:
    st.image(st.session_state.img_src, f'Generated image for "{st.session_state.user_img_prompt_input}"')
    st.session_state.curr_user_img_rating = st.slider(
      "How did we do? (1 = terrible, 10 = perfect)",
      min_value=1.0,
      max_value=10.0,
      step=0.1,
      value=st.session_state.curr_user_img_rating
    )
    st.button("Submit Rating", on_click=submitUserRatingHandler)
    st.write(f'Voting round {st.session_state.voting_rounds_completed+1} of {NUM_VOTING_ROUNDS}')

  ####################################### Session is over section #######################################
  if st.session_state.voting_rounds_completed == NUM_VOTING_ROUNDS:
    highest_rating = max(map(lambda obj: obj["rating"], st.session_state.user_img_ratings))
    highest_rated_imgs = list(filter(lambda obj: obj["rating"] == highest_rating, st.session_state.user_img_ratings))
    st.write(f'Hooray! Here {"are" if highest_rating != 1 else "is"} your highest rated {"images" if highest_rating != 1 else "image"}:')
    for obj in highest_rated_imgs:
      st.image(obj["img_src"], f'Generated image for "{st.session_state.user_img_prompt_input}", with a rating of {obj["rating"]}')
    st.balloons()

  for _ in range(15):
    st.write('')

  st.markdown("### In case you want to review your earlier images")
  for i in range(st.session_state.voting_rounds_completed):
    with st.expander(f'Image #{i + 1}'):
      user_img_rating = st.session_state.user_img_ratings[i]
      st.image(user_img_rating["img_src"], f'Generated image for "{st.session_state.user_img_prompt_input}", with a rating of {user_img_rating["rating"]}')

  for _ in range(15):
    st.write('')
  st.write("Made with :heart: by Michael Cooley and Ian Penrod")
