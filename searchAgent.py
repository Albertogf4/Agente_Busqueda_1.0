from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END # docs de StateGraph https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph
from langchain_openai import ChatOpenAI
import os
import operator
from IPython.display import display, Markdown



_ = load_dotenv()

# Search Tool
tool = TavilySearchResults(max_results=4, search_depth="advanced") # @TODO: number of results
#print(type(tool))
#print(tool.name)

# Agent State
class AgentState(TypedDict):
    """Información disponible en todo momento, con Annotated se pueden agregar metadatos sin perder información anterior"""
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, system_prompt=""):
        self.system = system_prompt # Prompt del sistema
        graph = StateGraph(AgentState) # Construiremos el grafo con los "nodes y edges" con acceso al estado del agente (lectura y escritura)
        # Nodos
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", 
            self.exists_action,
            {True: "action", False: END}
        )
        # Edges
        graph.add_edge("action", "llm")
        # Entry Point
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        # Tools
        self.tools = {t.name: t for t in tools} 
        # Model
        self.model = model.bind_tools(tools) # Enlazamos el modelo con las herramientas

    # Método para verificar si el llm quiere llamar a una función
    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    # Método para llamar al llm
    def call_llm(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    # Método para ejecutar las tools
    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls # Útimo mensaje, obtenido del estado
        results = []
        for t in tool_calls: # Iteramos sobre las llamadas a las herramientas para cuando haya más de una
            #print(f"Calling: {t}")
            if not t['name'] in self.tools:      # checkeo de error en la llamada
                #print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        #print("Back to the model!")
        return {'messages': results}
    
api_key = os.getenv("OPENAI_API_KEY")
prompt = """
Eres un investigador y analista empresarial experto.  
1. Utiliza el buscador para obtener información precisa y actualizada sobre la empresa solicitada.  
2. Realiza múltiples búsquedas de forma secuencial si es necesario, pero siempre optimizando las palabras clave para obtener resultados relevantes.  
3. Antes de realizar cada búsqueda, asegúrate de comprender exactamente qué información necesitas y enfoca tus consultas en ello.  
4. Consulta y analiza la información de manera crítica y estructurada.  
5. Proporciona siempre las fuentes específicas de donde obtuviste la información, indicando el enlace o los datos relevantes.  
6. Organiza tus respuestas de manera clara, diferenciando entre cada pregunta o aspecto abordado.  
7. Si recibes una petición muy larga o compleja, divídela en partes, haz busquedas de cada parte en orden y cuando tengas toda la información, responde a la pregunta completa.
"""

llm_model = ChatOpenAI(model="gpt-4o", api_key=api_key)  
abot = Agent(llm_model, [tool], system_prompt=prompt)


if __name__ == "__main__": 
    query = "¿Qué acción subió más en 2024 en USA? ¿Cual es el valor de dicha empresa a día de hoy?"
    messages = [HumanMessage(content=query)]
    result = abot.graph.invoke({"messages": messages})
    print(f"**Query:** {query}")
    print(f"**Result:** {result['messages'][-1].content}")
    