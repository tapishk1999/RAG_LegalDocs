"""
Ground-truth Q&A pairs for RAGAS evaluation.
Covers all 3 acts with factual questions that have verifiable answers.
"""

TEST_DATASET = [
    # ── Muslim Women (Protection of Rights on Marriage) Act, 2019 ──
    {
        "question": "What is the punishment for a husband who pronounces triple talaq?",
        "ground_truth": "A husband who pronounces triple talaq shall be punished with imprisonment for a term which may extend to three years, and shall also be liable to fine.",
        "act": "Muslim Women (Protection of Rights on Marriage) Act, 2019",
    },
    {
        "question": "Is triple talaq (talaq-e-biddat) declared void under the Muslim Women Act?",
        "ground_truth": "Yes. The Act declares any pronouncement of talaq by a Muslim husband upon his wife, by words spoken or written or in electronic form or in any other manner whatsoever, to be void and illegal.",
        "act": "Muslim Women (Protection of Rights on Marriage) Act, 2019",
    },
    {
        "question": "Can a Muslim woman receive subsistence allowance under the Muslim Women Act?",
        "ground_truth": "Yes. A Muslim woman upon whom talaq has been pronounced is entitled to receive subsistence allowance from her husband for herself and for her dependent children as determined by a Magistrate.",
        "act": "Muslim Women (Protection of Rights on Marriage) Act, 2019",
    },
    {
        "question": "Who can file a complaint under the Muslim Women Act?",
        "ground_truth": "A complaint can be filed by the married Muslim woman upon whom talaq has been pronounced, or any person related to her by blood or marriage.",
        "act": "Muslim Women (Protection of Rights on Marriage) Act, 2019",
    },
    {
        "question": "Can a Muslim woman claim custody of minor children under the Muslim Women Act?",
        "ground_truth": "Yes. A Muslim woman upon whom talaq is pronounced is entitled to the custody of her minor children, as may be determined by the Magistrate.",
        "act": "Muslim Women (Protection of Rights on Marriage) Act, 2019",
    },
    {
        "question": "What is the nature of the offence under the Muslim Women Act — cognizable or non-cognizable?",
        "ground_truth": "The offence is cognizable if information relating to the commission of such offence is given by the married Muslim woman upon whom talaq has been pronounced, or any person related to her by blood or marriage.",
        "act": "Muslim Women (Protection of Rights on Marriage) Act, 2019",
    },

    # ── Copyright Act, 1957 ──
    {
        "question": "What works are eligible for copyright protection under the Copyright Act 1957?",
        "ground_truth": "Copyright subsists in original literary, dramatic, musical and artistic works; cinematograph films; and sound recordings.",
        "act": "The Copyright Act, 1957",
    },
    {
        "question": "What is the term of copyright for a literary work under the Copyright Act 1957?",
        "ground_truth": "Copyright in a literary work subsists until sixty years from the beginning of the calendar year next following the year in which the author dies.",
        "act": "The Copyright Act, 1957",
    },
    {
        "question": "What is the definition of 'infringing copy' under the Copyright Act?",
        "ground_truth": "An infringing copy means, in relation to a literary, dramatic, musical or artistic work, a reproduction that is made or imported in contravention of the provisions of the Act.",
        "act": "The Copyright Act, 1957",
    },
    {
        "question": "Does the Copyright Act protect computer programs?",
        "ground_truth": "Yes. Computer programmes are included within the definition of 'literary work' under the Copyright Act, and thus enjoy copyright protection.",
        "act": "The Copyright Act, 1957",
    },
    {
        "question": "What are the moral rights of an author under the Copyright Act?",
        "ground_truth": "An author has the right to claim authorship of the work and to restrain or claim damages in respect of any distortion, mutilation, modification or other act in relation to the work which is done before the expiration of the copyright term, if such distortion would be prejudicial to the author's honour or reputation.",
        "act": "The Copyright Act, 1957",
    },
    {
        "question": "What is the penalty for copyright infringement under the Copyright Act?",
        "ground_truth": "Any person who knowingly infringes or abets infringement of copyright shall be punished with imprisonment for a term which shall not be less than six months but which may extend to three years and with a fine not less than fifty thousand rupees but which may extend to two lakh rupees.",
        "act": "The Copyright Act, 1957",
    },

    # ── Tribunals Reforms Act, 2021 ──
    {
        "question": "What is the term of office for the Chairperson of a Tribunal under the Tribunals Reforms Act 2021?",
        "ground_truth": "The Chairperson of a Tribunal shall hold office for a term of four years or until he attains the age of seventy years, whichever is earlier.",
        "act": "The Tribunals Reforms Act, 2021",
    },
    {
        "question": "What is the term of office for Members of a Tribunal under the Tribunals Reforms Act?",
        "ground_truth": "A Member of a Tribunal shall hold office for a term of four years or until he attains the age of sixty-seven years, whichever is earlier.",
        "act": "The Tribunals Reforms Act, 2021",
    },
    {
        "question": "Which body is responsible for recommending appointments to Tribunals under the 2021 Act?",
        "ground_truth": "A Search-cum-Selection Committee is responsible for recommending appointments to Tribunals under the Tribunals Reforms Act, 2021.",
        "act": "The Tribunals Reforms Act, 2021",
    },
    {
        "question": "Can a person who has served as Chairperson or Member of a Tribunal seek re-appointment?",
        "ground_truth": "A person who has held the office of Chairperson or Member shall not be eligible for re-appointment in the same Tribunal.",
        "act": "The Tribunals Reforms Act, 2021",
    },
    {
        "question": "What tribunals were abolished or merged under the Tribunals Reforms Act 2021?",
        "ground_truth": "The Act abolished certain existing tribunals including the Film Certification Appellate Tribunal and the Airports Appellate Tribunal, transferring their jurisdiction to High Courts or other bodies.",
        "act": "The Tribunals Reforms Act, 2021",
    },
    {
        "question": "Who heads the Search-cum-Selection Committee for tribunal appointments under the 2021 Act?",
        "ground_truth": "The Search-cum-Selection Committee is headed by the Chief Justice of India or a Judge of the Supreme Court nominated by the Chief Justice of India.",
        "act": "The Tribunals Reforms Act, 2021",
    },
    {
        "question": "What salary is provided to the Chairperson of a Tribunal under the Tribunals Reforms Act?",
        "ground_truth": "The salaries, allowances and service conditions of the Chairperson and Members of Tribunals are as specified in the Second Schedule of the Act.",
        "act": "The Tribunals Reforms Act, 2021",
    },
    {
        "question": "Under which ministry do Tribunals fall after the Tribunals Reforms Act 2021?",
        "ground_truth": "After the Tribunals Reforms Act 2021, Tribunals fall under the administrative control of the Ministry or Department of the Central Government which is responsible for the subject matter of the Tribunal.",
        "act": "The Tribunals Reforms Act, 2021",
    },
]
