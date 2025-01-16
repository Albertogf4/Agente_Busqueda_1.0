import os

class Config:
    """Configuración base para todos los entornos."""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "openai_secret")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "tavily_secret")
    DEBUG = False

class DevelopmentConfig(Config):
    """Configuración para el entorno de desarrollo."""
    DEBUG = True

class ProductionConfig(Config):
    """Configuración para el entorno de producción."""
    DEBUG = False

# Diccionario para seleccionar el entorno
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}
