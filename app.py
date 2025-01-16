from flask import Flask, render_template, request, jsonify
from searchAgent import Agent, TavilySearchResults, ChatOpenAI, HumanMessage
import os
from dotenv import load_dotenv
from config import config

load_dotenv()
env = os.getenv("FLASK_ENV", "production")

app = Flask(__name__)
app.config.from_object(config[env])

# Initialize the AI agent
tool = TavilySearchResults(max_results=5, max_tokens=400, search_depth="basic") # search_depth, topic, days, max_results, include_domains, exclude_domain
prompt = """Eres un experto investigador y analista de empresas.  
Usa el buscador para obtener información de la empresa de la que se te pregunta. 
Tienes permitido hacer múltiples llamadas (a la vez o de forma secuencial). 
Usa palabras clave cuando uses tavily_search_results_json para cada búsqueda
Solo busca información de lo que estés seguro que quieres buscar. 
"""

api_key = os.getenv("OPENAI_API_KEY")
llm_model = ChatOpenAI(model="gpt-4o", api_key=api_key)
abot = Agent(llm_model, [tool], system_prompt=prompt)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.json['query']
    messages = [HumanMessage(content=query)]
    result = abot.graph.invoke({"messages": messages})
    response = result['messages'][-1].content
    return jsonify({'result': response})

if __name__ == '__main__':
    app.run(debug=True)