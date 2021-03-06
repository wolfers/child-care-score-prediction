'''
Dictionaries to map the ccqb ratings to the official ratings.
The data is pretty dirty so there is a lot of repeated columns and typos.

missing or unknown columns in the ers official ratings (likely drop them)
 'General supervision of children'
 'Display for children'
'''

ers_dict = {"Indoor Space - Score": ["Indoor space used for child care",  'Indoor space'],
            "Furn. care, play, learning - Score": ["Furniture for routine care and play", "Furniture for routine care, play & learning", 'Furniture for routine care, play, and learning'],
            "Furnishings for relax, comfort - Score": ["Furnishings for relaxation", 'Provision for relaxation and comfort6', 'Provision for relaxation and comfort'],
            "Room arrangement for play - Score": ["Room arrangement for play", 'Room arrangement', 'Arrangement for space for child care'],
            "Space for privacy - Score": ['Space for privacy'],
            "Child-related display - Score": ['Child-related display', 'Display for children5'],
            'Space for gross motor play - Score': ['Space for gross motor'],
            'Gross motor equipment - Score': ['Gross motor equipment'],
            'Greeting/departing - Score': ['Greeting/departing'],
            'Meals/snacks - Score': ['Meals/snacks'],
            'Nap/rest - Score': ['Nap/rest', 'Nap'],
            'Toileting/diapering - Score': ['Toileting/diapering', 'Diapering/toileting'],
            'Health practices - Score': ['Health practices'],
            'Safety practices - Score': ['Safety practices'],
            'Helping children understand lang - Score': ['Helping children understand language'],
            'Helping children use language - Score': ['Helping children use language'],
            'Books and pictures - Score': ['Books and pictures', 'Using books'],
            'Encouraging child to communicate - Score': ['Encouraging children to communicate'],
            'Using lang to dev. reason skills - Score': ['Using language to develop reasoning skills'],
            'Informal use of language - Score': ['Informal use of language'],
            'Fine motor - Score': ['Fine motor'],
            'Art - Score': ['Art'],
            'Music/movement - Score': ['Music and movement', 'Music/movement'],
            'Blocks - Score': ['Blocks'],
            'Dramatic play - Score': ['Dramatic play'],
            'Math/number - Score': ['Math/number'],
            'Nature/science - Score': ['Nature/science'],
            'Sand/Water - Score': ['Sand/water', 'Sand and water play'],
            'Use of TV, video, computers - Score': ['Use of TV, video, and/or computer', 'Use of TV, video, and/or computers'],
            'Prom. acceptance of diversity - Score': ['Promoting acceptance of diversity'],
            'Active physical play - Score': ['Active physical play'],
            'Supervision Play & Learning - Score': ['Supervision of play and learning'],
            'Staff-child interactions - Score': ['Staff-child interactions', 'Staff-child interaction', 'Provider-child interaction'],
            'Discipline - Score': ['Discipline'],
            'Interactions among children - Score': ['Interactions among children', 'Peer interaction'],
            'Super. of gross motor activities - Score': ['Supervision of gross motor activities'],
            'Schedule - Score': ['Schedule'],
            'Free Play - Score': ['Free play'],
            'Group Time - Score': ['Group time', 'Group play activities'],
            'Prov. for child with disability - Score': ['Provisions for children with disabilities']}


class_dict = {'Emotional and Behavioral Support Score': ['Emotional (and Behavioral*) Support_(Negative Affect, Punitive Control, Sarcasm/Disrespect)',
                                                        'Emotional (and Behavioral*) Support_Behavior Management and Guidance',
                                                        'Emotional (and Behavioral*) Support_Positive Climate/Relational Climate',
                                                        'Emotional (and Behavioral*) Support_Regard for Student Perspectives (Student Focus, Flexibility, Leadership/Independence)',
                                                        'Emotional (and Behavioral*) Support_Teacher Sensitivity(Including Infant)',
                                                        'Emotional and Behavioral Support_Behavior Guidance (Proactive, Supporting positive behavior, Problem behavior)',
                                                        'Emotional and Behavioral Support_Negative Climate (Negative Affect, Punitive Control, Sarcasm/Disrespect, Severe Negativity)',
                                                        'Emotional and Behavioral Support_Positive Climate (Relationships, Positive Affect, Positive Communication, Respect)',
                                                        'Emotional and Behavioral Support_Regards for Child Perspectives (Child focus, Flexibility, Support of independance)',
                                                        'Emotional and Behavioral Support_Teacher Sensitivity (Awareness, Responsiveness, Addresses Problems, Student Control)',
                                                        'Emotional Support_Negative Climate (Negative Affect, Punitive Control, Sarcasm/Disrespect, Severe Negativity)2',
                                                        'Emotional Support_Positive Climate (Relationships, Positive Affect, Positive Communication, Respect)3',
                                                        'Emotional Support_Regard for Student Perspectives (Flexibility and Student Focus, Support for Autonomy and Leadership, Student Expression, Restriction of Movement)',
                                                        'Emotional Support_Teacher Sensitivity (Awareness, Responsiveness, Addresses Problems, Student Control)4'],
              'Classroom Organization Score': ['Classroom Organization_Behavior Management (Clear Behavior Expectations, Proactive, Redirection of Misbehavior, Student Behavior)',
                                               'Classroom Organization_Instructional Learning Formats (Effective Facilitation, Variety of Modalities and Materials, Student Interest, Clarity of Learning Objectives)',
                                               'Classroom Organization_Productivity (Maximizing Learning Time, Routines, Transitions, Preparation)',
                                               'Classroom Organization_Produtivity (Maximizing Learning Time, Routines, Transitions, Preparations)'],
              'Instructional Support Score': ['Instructional Support_Concept Development (Analysis and Reasoning, Creating, Integration, Connections to the Real World)',
                                            'Instructional Support_Language Modeling (Frequent Conversation, Open-Ended Questions, Repetition and Extension, Self and Parallel Talk, Advanced Language)',
                                            'Instructional Support_Quality of Feedback (Scaffolding, Feedback Loops, Prompting Thought Processes, Providing Information, Encouragement and Affirmation)',
                                            'Instructional Support/Engaged Support for Learning_Concept Development (Analysis and Reasoning, Creating, Integration, Connections to the Real World)5',
                                            'Instructional Support/Engaged Support for Learning_Facilitation of Learning and Development/Facilitated exploration',
                                            'Instructional Support/Engaged Support for Learning_Language   Modeling/Early Language Support',
                                            'Instructional Support/Engaged Support for Learning_Quality of Feedback (Scaffolding, Provides Information, Feedback Loops, Encouragement/Affirmation)6'],
              'Engaged Support for Learning Score': ["Engaged Support for Learning_Facilitation of Learning and Development (Active facilitation, Expansion of Cognition, Children's active engagement)",
                                                    'Engaged Support for Learning_Language Modeling (Supporting language use, Repetition and extension)',
                                                    'Engaged Support for Learning_Quality of Feedback (Scaffolding, Providing Information, Encouragement of affirmation)']}