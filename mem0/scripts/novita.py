from llms.novita import NovitaLLM


def main():
    llm = NovitaLLM()
    print(llm.generate_response([{"role": "user", "content": "Hello, how are you?"}]))


if __name__ == "__main__":
    main()
