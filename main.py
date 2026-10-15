
from retriver.retriver import retrieve_posts 
from analysis.report import generate_report 

def main():

    query = input("Enter your query: ")

    print("\nRetrieving relevant posts...")

    documents = retrieve_posts(query,20)

    if not documents:
        print("No relevant posts found.")
        return

    print(f"Retrieved {len(documents)} posts.")

    print("\nGenerating intelligence report...")

    

  
    
    for chunk in generate_report(query, documents):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    main()
