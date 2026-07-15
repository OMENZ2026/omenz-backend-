import random

# ==========================================================
# OMENZ ROUTER v0.1
# ==========================================================

OPENAI = "openai"
ANTHROPIC = "anthropic"
PERPLEXITY = "perplexity"


def route_task(task_type: str):
    """
    Basic OMENZ router.

    Returns which provider should handle a task.
    """

    task_type = task_type.lower()

    routes = {
        "chat": OPENAI,
        "code": OPENAI,
        "reasoning": ANTHROPIC,
        "analysis": ANTHROPIC,
        "research": PERPLEXITY,
        "search": PERPLEXITY,
    }

    return routes.get(task_type, OPENAI)


def provider_name(provider):
    names = {
        OPENAI: "PENI",
        ANTHROPIC: "AUDE",
        PERPLEXITY: "XITY",
    }

    return names.get(provider, "UNKNOWN")


def health_check():
    return {
        OPENAI: True,
        ANTHROPIC: True,
        PERPLEXITY: True,
    }


def choose_available(task_type):
    preferred = route_task(task_type)

    health = health_check()

    if health.get(preferred):
        return preferred

    for provider, status in health.items():
        if status:
            return provider

    return None


def random_provider():
    return random.choice(
        [
            OPENAI,
            ANTHROPIC,
            PERPLEXITY,
        ]
    )


if __name__ == "__main__":

    task = "research"

    provider = choose_available(task)

    print(
        f"Task: {task} -> {provider_name(provider)} ({provider})"
    )
