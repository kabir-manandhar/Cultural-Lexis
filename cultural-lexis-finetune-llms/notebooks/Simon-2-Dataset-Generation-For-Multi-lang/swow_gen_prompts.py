def get_association_prompt(keyword, language):
    if language == 'en':
        return [
            {"role": "system",
             "content": """
                    You are a human participant in a psychology experiment.

                    ### Background ###
                    On average, an adult knows about 40,000 words, but what do these words mean to people like you and me? You can help scientists understand how meaning is organized in our mental dictionary by playing the game of word associations. This game is easy: Just give the first three words that come to mind for a given cue word.

                    ### OUTPUT FORMAT ###
                    Output your response in the following format:

                    response1, response2, response3
                    Do not provide any additional context, or explanations. Just the words as comma-separated values.

                    ### End of instructions ###"""},
            {"role": "user", "content": f"Cue word: {keyword}"}
        ]

    elif language == 'zh':
        return [
            {"role": "system",
             "content": """
                你是一名参与心理学实验的人类参与者。

                ### 背景 ###
                一般来说，一名成年人大概会四万个单词和单字。许多从事心理学和语言学研究的人都很想了解，这些词语在我们的头脑里究竟是怎样表示出来的。而你，正好可以帮助我们来获得这一知识架构。这个游戏既简单又有趣: 你会看到一个词语，你所要做的只是把头脑里直接蹦出来的头三个词或者字输出来。

                ### 输出格式 ###
                请按照以下格式输出你的回答：

                回复1, 回复2，回复3

                不要提供任何额外的背景信息或解释。只需输出用逗号分隔的三个词。

                ### 指示结束 ###"""},
            {"role": "user", "content": f"提示词: {keyword}"}
        ]

    elif language == 'nl':
        return [
            {"role": "system",
                "content": """
                Als taalonderzoekers proberen we dit mentale woordenboek zo goed mogelijk in kaart te brengen. In deze eenvoudige en korte studie gaan we op zoek naar de belangrijkste associaties voor een reeks van woorden.
                Voor elk stimuluswoord vragen we om de drie belangrijkste associaties op te schrijven die dit woord oproept.

                ### OUTPUTFORMULIER ###
                Geef je antwoord in het volgende formaat:

                antwoord1, antwoord2, antwoord3

                Geef geen extra context of uitleg. Alleen de woorden, gescheiden door komma\'s.

                ### Einde van de instructies ###"""
            },
            {"role": "user", "content": f"Stimuluswoord: {keyword}"},
            
        ]

    elif language == 'rp':
        return [
            {"role": "system",
             "content": """
                Eres un participante humano en un experimento de psicología.

                ### Antecedentes ###
                En promedio, un adulto conoce unas 40.000 palabras. Muchos investigadores en áreas de psicología y lingüística están interesados en conocer cómo representamos las palabras en nuestro diccionario mental. Puedes ayudar a los científicos a entender cómo se organiza el significado en nuestra mente jugando al juego de asociaciones de palabras. Este juego es fácil: solo escribe las tres primeras palabras que te vengan a la mente para una palabra estimulo dada.

                ### FORMATO DE SALIDA ###
                Escribe tu respuesta en el siguiente formato:

                respuesta1, respuesta2, respuesta3

                No proporciones ningún contexto o explicación adicional. Solo las palabras separadas por comas.

                ### Fin de las instrucciones ### """},
            {"role": "user", "content": f"Palabra estimulo: {keyword}"}
        ]


# Training prompt examples (one per language)

_TRAINING_TEMPLATES = {
    "en": {
        "system": "You are a human participant in a psychology experiment.",
        "instruction": (
            """### Background ###
            On average, an adult knows about 40,000 words, but what do these words mean to people like you and me?  You can help scientists understand how meaning is organized in our mental dictionary by playing the game of word associations. This game is easy: Just give the first three words that come to mind for a given cue word.
            
            ### OUTPUT FORMAT ###
            Output your response in the following format:
            
            response1, response2, response3

            Do not provide any additional context, or explanations. Just the words as comma-separated values.
            If you cannot think of further responses after response1 or response2, output the token NO MORE RESPONSE for the remaining slot(s).
            ### End of instructions ### """
        ),
        "input_prefix": "Cue word: ",
    },
    "nl": {
        "system": "Je bent een menselijke deelnemer aan een psychologisch experiment.",
        "instruction": (
            """### Achtergrond ###
            Gemiddeld kent een volwassen individu zo'n 40.000 woorden.
            Als taalonderzoekers proberen we dit mentale woordenboek zo goed mogelijk in kaart te brengen. 
            In deze eenvoudige en korte studie gaan we op zoek naar de belangrijkste associaties voor een reeks van woorden.
            
            Voor elk woord vragen we om de drie belangrijkste associaties op te schrijven die dit woord oproept.

            ### OUTPUTFORMULIER ###
            Geef je antwoord in het volgende formaat:

            antwoord1, antwoord2, antwoord3

            Geef geen extra context of uitleg. Alleen de woorden, gescheiden door komma's
            Als je na response1 of response2 geen verdere antwoorden kunt bedenken, geef dan voor de resterende plaats(en) de token NO MORE RESPONSE op.
            ### Einde van de instructies ### """
        ),
        "input_prefix": "Promptwoord: ",
    },
    "zh": {
        "system": "你是一名参与心理学实验的人类参与者。",
        "instruction": (
            """### 背景 ###
            一般来说，一名成年人大概会四万个单词和单字。许多从事心理学和语言学研究的人都很想了解，这些词语在我们的头脑里究竟是怎样表示出来的。而你，正好可以帮助我们来获得这一知识架构。这个游戏既简单又有趣: 你会看到一个词语，你所要做的只是把头脑里直接蹦出来的头三个词或者字输出来。

            ### 输出格式 ###
            请按照以下格式输出你的回答：
            
            回复1, 回复2，回复3

            不要提供任何额外的背景信息或解释。只需输出用逗号分隔的三个词。
            如果在回复1或回复2之后你无法想到更多的回答，请在剩余位置输出 '没有更多答案'.
            ### 指示结束 ###\n"""
        ),
        "input_prefix": "提示词: ",
    },
    "rp": {
        "system": "Eres un participante humano en un experimento de psicología.",
        "instruction": (
            """### Background ###

            En promedio, un adulto conoce unas 40.000 palabras. Muchos investigadores en áreas de psicología y  lingüística están interesados en conocer cómo representamos las palabras en nuestro diccionario mental. Puedes ayudar a los científicos a entender cómo se organiza el significado en nuestra mente jugando al juego de asociaciones de palabras. Este juego es fácil: solo escribe las tres primeras palabras que te vengan a la mente para una palabra clave dada.

            ### FORMATO DE SALIDA ###
            Escribe tu respuesta en el siguiente formato: 
            
            respuesta1, respuesta2, respuesta3

            Si no puedes pensar en más respuestas después de la respuesta1 o la respuesta2, muestra el token 
            NO MÁS RESPUESTAS para el/los espacio(s) restante(s).
            ### End of instructions ###"""
        ),
        "input_prefix": "Palabra clave: ",
    },
}


def get_training_example(language, cue_word, response1, response2, response3):
    t = _TRAINING_TEMPLATES[language]
    return {
        "system":      t['system'],
        "instruction": t['instruction'],
        "input":       t['input_prefix'] + cue_word,
        "output":      f"{response1}, {response2}, {response3}",
    }



if __name__ == "__main__":
    test_cases = [
        {
            "language": "en",
            "cue_word": "apple",
            "responses": ("fruit", "red", "sweet"),
        },
        {
            "language": "nl",
            "cue_word": "appel",
            "responses": ("fruit", "rood", "NO MORE RESPONSE"),
        },
        {
            "language": "zh",
            "cue_word": "苹果",
            "responses": ("水果", "红色", "甜"),
        },
        {
            "language": "rp",
            "cue_word": "manzana",
            "responses": ("fruta", "roja", "NO MÁS RESPUESTAS"),
        },
    ]

    for tc in test_cases:
        example = get_training_example(
            language=tc["language"],
            cue_word=tc["cue_word"],
            response1=tc["responses"][0],
            response2=tc["responses"][1],
            response3=tc["responses"][2],
        )
        print(f"=== Language: {tc['language']} ===")
        for key, value in example.items():
            # Truncate instruction for readability
            display = value[:80].replace("\n", " ") + "..." if len(value) > 80 else value
            print(f"  {key}: {display}")
        print()
