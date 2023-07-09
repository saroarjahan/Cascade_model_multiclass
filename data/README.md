# Dataset Format #

**Note**: This Dataset is derived from the original [HateXplain](https://github.com/hate-alert/HateXplain) dataset

Sample Entry:

~~~json
{
  "24198545_gab": {
    "post_id": "24198545_gab",
    "annotators": [
      {
        "label": "hatespeech",
        "annotator_id": 4,
        "target": ["African"]
      },
      {
        "label": "hatespeech",
        "annotator_id": 3,
        "target": ["African"]
      },
      {
        "label": "offensive",
        "annotator_id": 5,
        "target": ["African"]
      }
    ],
    "rationales":[
    [0,0,0,0,0,0,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ],
    "post_tokens": ["and","this","is","why","i","end","up","with","nigger","trainee","doctors","who","can","not","speak","properly","lack","basic","knowledge","of","biology","it","truly","scary","if","the","public","only","knew"]
  }
}
~~~

:small_blue_diamond:post_id : Unique id for each post

:small_blue_diamond:annotators : The list of annotations from each annotator

:small_blue_diamond:annotators[label] : The label assigned by the annotator to this post. Possible values: [Hatespeech, Offensive, Normal]

:small_blue_diamond:annotators[annotator_id] : The unique Id assigned to each annotator

:small_blue_diamond:annotators[target] : A list of target community present in the post

:small_blue_diamond:rationales : A list of rationales selected by annotators. Each rationales represents a list with values 0 or 1. A value of 1 means that the token is part of the rationale selected by the annotator. To get the particular token, we can use the same index position in "post_tokens"

:small_blue_diamond:post_tokens : The list of tokens representing the post which was annotated

## Post ids divisions ##

[Post_id_divisions](https://bitbucket.org/adrian-secteam/code-test/src/main/data/post_id_divisions.json) has a dictionary having train, valid and test post ids that are used to divide the dataset into train, val and test set in the ratio of 8:1:1.
