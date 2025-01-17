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
tool = TavilySearchResults(max_results=5, max_tokens=500, search_depth="advanced") # search_depth, topic, days, max_results, include_domains, exclude_domain
prompt =  """
Eres un investigador y analista empresarial experto.  
1. Utiliza el buscador para obtener información precisa y actualizada sobre la empresa solicitada.  
2. Realiza múltiples búsquedas de forma secuencial si es necesario, pero siempre optimizando las palabras clave para obtener resultados relevantes.  
3. Antes de realizar cada búsqueda, asegúrate de comprender exactamente qué información necesitas y enfoca tus consultas en ello.  
4. Consulta y analiza la información de manera crítica y estructurada.  
5. Proporciona siempre las fuentes específicas de donde obtuviste la información, indicando el enlace o los datos relevantes.  
6. Organiza tus respuestas de manera clara, diferenciando entre cada pregunta o aspecto abordado.  
7. Si recibes una petición muy larga o compleja, divídela en partes, haz busquedas de cada parte en orden y cuando tengas toda la información, responde a la pregunta completa.
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