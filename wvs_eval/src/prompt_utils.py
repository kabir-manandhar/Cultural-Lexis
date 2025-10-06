def create_system_prompt(country_name: str) -> str:
    """
    Creates the system prompt for survey questions, customized for the given country.

    Args:
        country_name (str): Name of the country ("United States" or "China").

    Returns:
        str: A formatted system prompt in the appropriate language.
    """
    if country_name == "China":
        system_prompt = (
            "我们正在进行一项有关社会治理的调查研究。这项研究的目的，是了解和考察新型城镇化阶段本区域城乡居民对社会治理各个方面的认知、评价和态度，以及公众的协商参与情况，为推进本区域新型城镇化建设提供真实可靠的研究素材。 非常感谢您接受我们的采访。您的看法、意见不仅对这项研究非常重要，而且也能为本 区域新型城镇化建设提供积极的推动作用。您的回答没有对错之分，我们需要的是您的真实 想法。这项研究试图采用科学的、定量的社会调查方法，在采访过程中，也许会有一些您所 不熟悉的提问方式，到时我们会向您解释。这次采访是匿名的，我们将按照《统计法》的规 定严格保密，相关信息只用于统计分析，保证不在任何时间、任何情况下，以直接或间接方式提到您本人。谢谢您的合作，现在就让我们开始吧"
            "请确保你的输出符合以下确切格式：\n"
            "预测 - <类别> | 原因 - <原因>"
        )
    elif country_name == "United States":
        system_prompt = (
            "We are carrying out a global study of what people value in life. This study will interview samples representing most of the world's people. Your name has been selected at random as part of a representative sample of the people in United States. I'd like to ask your views on a number of different subjects. Your input will be treated strictly confidential, but it will contribute to a better understanding of what people all over the world believe and want out of life."
            "Ensure your output follows this exact format:\n"
            "prediction - <category> | reason - <reason>"
        )
    else:
        raise ValueError(f"Unsupported country: {country_name}")

    return system_prompt
