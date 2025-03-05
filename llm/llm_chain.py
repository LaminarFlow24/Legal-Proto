import re
import google.generativeai as genai

from utils.env_util import EnvironmentVariables

class LLM:
    def __init__(self) -> None:
        env = EnvironmentVariables()
        # Configure Gemini API with your API key
        genai.configure(api_key=env.GOOGLE_API_KEY)
        # You can store the model name (or use it directly in summarize)
        self.model_name = 'gemini-2.0-flash'
        self.temperature = 0.7

    def get_chain(self):
        # Return a dummy chain or None if not required.
        return None

    @staticmethod
    def token_nums(api_response, data: str):
        # Your token calculation logic if needed.
        pass

    def get_prompt(self, relevant_documents: list, clause: str) -> str:
        # Create a prompt for the Gemini API
        prompt = (
            "You are a lease clause summarization expert. Your job is to Summarize the lease clauses "
            "from the below mentioned Context in a single and short paragraph. The final answer should "
            "fulfill the definition and the given guidelines for each clause. \n\n"
            "If the context has relevant information regarding the clause section, page number, clause number, etc. "
            "please add them at the end of the summary inside '()'. \n\n"
            "If the summarized answer is not relevant to the '{clause}', just output 'Clause Not Found' as the summary. "
            "Don't make your answer on your own. \n\n"
            "The given Context:{context}\n\n"
            "The Clause is {clause}.\n\n"
            "Answer:"
        ).format(context=" ".join(relevant_documents), clause=clause)
        return prompt

    def summarize(self, relevant_documents: list, query: str, chain=None, class_name=None) -> str:
        prompt = self.get_prompt(relevant_documents, query)
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            s = response.text
        except Exception as e:
            s = f"Error generating summary: {str(e)}"
        
        # Postprocess the response similar to your previous logic:
        res = re.sub(r"<.*?>", "", s)
        res = res.replace("\n", " ")
        not_found = re.findall(r"Clause Not Found", res)
        if not_found:
            output = not_found[0]
        else:
            output = res

        code_pattern_match = re.search(r"^(.*?)```", output)
        if code_pattern_match:
            output = code_pattern_match.group(1)

        quote_pattern_match = re.search(r'^(.*?)"""', output)
        if quote_pattern_match:
            output = quote_pattern_match.group(1)

        single_quote_pattern_match = re.search(r"^(.*?)'''", output)
        if single_quote_pattern_match:
            output = single_quote_pattern_match.group(1)

        return output