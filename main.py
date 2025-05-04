import sys
from agent import GaiaGeminiAgent

if __name__ == "__main__":
    try:
        agent = GaiaGeminiAgent()
    except ValueError as e:
        print(f"Initialization Error: {e}")
        print("Please ensure your GOOGLE_API_KEY is set in the .env file.")
        sys.exit(1)
    except Exception as e:
         print(f"An unexpected error occurred during initialization: {e}")
         sys.exit(1)


    print("\n--- GAIA-Style Gemini Agent ---")
    print("Enter your complex question (or type 'quit' to exit).")
    print("Example: Which of the fruits shown in the 2008 painting “Embroidery from Uzbekistan” were served as part of the October 1949 breakfast menu for the ocean liner that was later used as a floating prop for the film “The Last Voyage”? Give the items as a comma-separated list, ordering them in clockwise order based on their arrangement in the painting starting from the 12 o’clock position. Use the plural form of each fruit.")

    while True:
        user_query = input("\nYour Question: ")
        if user_query.lower() == 'quit':
            break
        if not user_query:
            continue

        print("\n> Processing query...")
        try:
            # Run the agent's process
            final_result = agent.run(user_query)

            print("\n======== Agent Final Answer ========")
            print(final_result)

            # Optionally print intermediate steps/history for debugging
            # print("\n======== Agent Internals (Debug) ========")
            # print("--- History ---")
            # for turn in agent.history:
            #     print(f"{turn['role'].upper()}: {turn['content']}")
            # print("\n--- Intermediate Results ---")
            # for key, value in agent.intermediate_results.items():
            #      print(f"- {key}: {value}")


        except Exception as e:
            print(f"\nAn unexpected error occurred during agent execution: {e}")
            # You might want to print traceback for debugging
            # import traceback
            # traceback.print_exc()

    print("\n--- Agent session ended ---")