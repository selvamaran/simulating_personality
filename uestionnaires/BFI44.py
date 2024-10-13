# questionnaires/BFI44.py

# SCORES_DICT for BFI-44 (User can modify or extend this dictionary for other questionnaires)
SCORES_DICT = {
    "disagree strongly with the statement": 1,
    "disagree a little with the statement": 2,
    "agree nor disagree with the statement": 3,
    "agree a little with the statement": 4,
    "agree strongly with the statement": 5
}

# Questions for the BFI-44 questionnaire (User can replace or add new questionnaires)
QUESTIONS = {
    'openness':
      [
          {
            'q_num': 5,
            'q_statement': 'I see myself as someone who Is original, comes up with new ideas',
            'q_type': 'direct'
          },
          {
            'q_num': 10,
            'q_statement': 'I see myself as someone who Is curious about many different things',
            'q_type': 'direct'
          },
          {
            'q_num': 15,
            'q_statement': 'I see myself as someone who Is ingenious, a deep thinker',
            'q_type': 'direct'
          },
          {
            'q_num': 20,
            'q_statement': 'I see myself as someone who Has an active imagination',
            'q_type': 'direct'
          },
          {
            'q_num': 25,
            'q_statement': 'I see myself as someone who Is inventive',
            'q_type': 'direct'
          },
          {
            'q_num': 30,
            'q_statement': 'I see myself as someone who Values artistic, aesthetic experiences',
            'q_type': 'direct'
          },
          {
            'q_num': 35,
            'q_statement': 'I see myself as someone who Prefers work that is routine',
            'q_type': 'inverted'
          },
          {
            'q_num': 40,
            'q_statement': 'I see myself as someone who Likes to reflect, play with ideas',
            'q_type': 'direct'
          },
          {
            'q_num': 41,
            'q_statement': 'I see myself as someone who Has few artistic interests',
            'q_type': 'inverted'
          },
          {
            'q_num': 44,
            'q_statement': 'I see myself as someone who Is sophisticated in art, music, or literature',
            'q_type': 'direct'
          },
      ],
    'conscientiousness':
      [
          {
            'q_num': 3,
            'q_statement': 'I see myself as someone who Does a thorough job',
            'q_type': 'direct'
          },
          {
            'q_num': 8,
            'q_statement': 'I see myself as someone who Can be somewhat careless',
            'q_type': 'inverted'
          },
          {
            'q_num': 13,
            'q_statement': 'I see myself as someone who Is a reliable worker',
            'q_type': 'direct'
          },
          {
            'q_num': 18,
            'q_statement': 'I see myself as someone who Tends to be disorganized',
            'q_type': 'inverted'
          },
          {
            'q_num': 23,
            'q_statement': 'I see myself as someone who Tends to be lazy',
            'q_type': 'inverted'
          },
          {
            'q_num': 28,
            'q_statement': 'I see myself as someone who Perseveres until the task is finished',
            'q_type': 'direct'
          },
          {
            'q_num': 33,
            'q_statement': 'I see myself as someone who Does things efficiently',
            'q_type': 'direct'
          },
          {
            'q_num': 38,
            'q_statement': 'I see myself as someone who Makes plans and follows through with them',
            'q_type': 'direct'
          },
          {
            'q_num': 43,
            'q_statement': 'I see myself as someone who Is easily distracted',
            'q_type': 'inverted'
          }
      ],
    'extraversion':
      [
          {
            'q_num': 1,
            'q_statement': 'I see myself as someone who Is talkative',
            'q_type': 'direct'
          },
          {
            'q_num': 6,
            'q_statement': 'I see myself as someone who Is reserved',
            'q_type': 'inverted'
          },
          {
            'q_num': 11,
            'q_statement': 'I see myself as someone who Is full of energy',
            'q_type': 'direct'
          },
          {
            'q_num': 16,
            'q_statement': 'I see myself as someone who Generates a lot of enthusiasm',
            'q_type': 'direct'
          },
          {
            'q_num': 21,
            'q_statement': 'I see myself as someone who Tends to be quiet',
            'q_type': 'inverted'
          },
          {
            'q_num': 26,
            'q_statement': 'I see myself as someone who Has an assertive personality',
            'q_type': 'direct'
          },
          {
            'q_num': 31,
            'q_statement': 'I see myself as someone who Is sometimes shy, inhibited',
            'q_type': 'inverted'
          },
          {
            'q_num': 36,
            'q_statement': 'I see myself as someone who Is outgoing, sociable',
            'q_type': 'direct'
          }
      ],
    'agreeableness':
      [
          {
            'q_num': 2,
            'q_statement': 'I see myself as someone who Tends to find fault with others',
            'q_type': 'inverted'
          },
          {
            'q_num': 7,
            'q_statement': 'I see myself as someone who Is helpful and unselfish with others',
            'q_type': 'direct'
          },
          {
            'q_num': 12,
            'q_statement': 'I see myself as someone who Starts quarrels with others',
            'q_type': 'inverted'
          },
          {
            'q_num': 17,
            'q_statement': 'I see myself as someone who Has a forgiving nature',
            'q_type': 'direct'
          },
          {
            'q_num': 22,
            'q_statement': 'I see myself as someone who Is generally trusting',
            'q_type': 'direct'
          },
          {
            'q_num': 27,
            'q_statement': 'I see myself as someone who Can be cold and aloof',
            'q_type': 'inverted'
          },
          {
            'q_num': 32,
            'q_statement': 'I see myself as someone who Is considerate and kind to almost everyone',
            'q_type': 'direct'
          },
          {
            'q_num': 37,
            'q_statement': 'I see myself as someone who Is sometimes rude to others',
            'q_type': 'inverted'
          },
          {
            'q_num': 42,
            'q_statement': 'I see myself as someone who Likes to cooperate with others',
            'q_type': 'direct'
          }
      ],
    'neuroticism':
      [
          {
            'q_num': 4,
            'q_statement': 'I see myself as someone who Is depressed, blue',
            'q_type': 'direct'
          },
          {
            'q_num': 9,
            'q_statement': 'I see myself as someone who Is relaxed, handles stress well',
            'q_type': 'inverted'
          },
          {
            'q_num': 14,
            'q_statement': 'I see myself as someone who Can be tense',
            'q_type': 'direct'
          },
          {
            'q_num': 19,
            'q_statement': 'I see myself as someone who Worries a lot',
            'q_type': 'direct'
          },
          {
            'q_num': 24,
            'q_statement': 'I see myself as someone who Is emotionally stable, not easily upset',
            'q_type': 'inverted'
          },
          {
            'q_num': 29,
            'q_statement': 'I see myself as someone who Can be moody',
            'q_type': 'direct'
          },
          {
            'q_num': 34,
            'q_statement': 'I see myself as someone who Remains calm in tense situations',
            'q_type': 'inverted'
          },
          {
            'q_num': 39,
            'q_statement': 'I see myself as someone who Gets nervous easily',
            'q_type': 'direct'
          }
      ]
}
