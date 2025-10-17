# learns using no labeled examples, uses other info to learn labels from data
# learns three vars, the input x, the label y and the random var which describes the task T
# prob distr is then p(y|x, T)
# allows us the classify text with minimal training examples
# note numpy 1.10 used here

# we use TARS (task aware representation of sentences) - load pre-trained model below to use
# for text classification:
from flair.models import TARSClassifier
from flair.models import TARSTagger
from flair.data import Sentence

# first we use zero shot learning for text classifying - classifies input sentence under a category using the model already built - no other features needed so zero shot
def zero_shot_text_classify():
    # load data set
    tars: TARSClassifier = TARSClassifier.load('tars-base')

    # our input sentence to classify and input classes to choose prediction from
    sentence = Sentence(
        "The 2020 United States presidential election was the 59th quadrennial presidential election, held on Tuesday, November 3, 2020")
    classes = ["sports", "politics", "science", "art"]

    # predict class of input sentence
    tars.predict_zero_shot(sentence, classes)

    # print sentence after classifying
    print("\n", sentence)

# zero_shot_text_classify()


# can also do zero shot for named entity recognition
# takes input sentence and applies all relevant classes to it that are applicable
# e.g. the sentence “Mark Zuckerberg is one of the founders of Facebook, a company from the United States” = three types of entities: “Person”: Mark Zuckerberg. “Company”: Facebook. “Location”: United States.
# takes tokenized words and identifies keywords which it classifies
def zero_shot_NER():
    # load ner tagger
    tars = TARSTagger.load('tars-ner')

    # define some input sentences to classify
    sentences = [
        Sentence("The Humboldt University of Berlin is situated near the Spree in Berlin, Germany"),
        Sentence("Bayern Munich played against Real Madrid"),
        Sentence("I flew with an Airbus A380 to Peru to pick up my Porsche Cayenne"),
        Sentence("Game of Thrones is my favorite series"),
    ]

    # define some classes to use as labels = used to name out entities in input sentences
    labels = ["Soccer Team", "University", "Vehicle", "River", "City", "Country", "Person", "Movie", "TV Show"]
    tars.add_and_switch_to_new_task('task 1', labels, label_type='ner')   # ensures ner task is applied

    # predict classes for each sentence above by applying model
    for sentence in sentences:
        tars.predict(sentence)
        print(sentence.to_tagged_string("ner"))

zero_shot_NER()


