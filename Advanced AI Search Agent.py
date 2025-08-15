import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage
from langchain.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import datetime
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
import re

# Page configuration
st.set_page_config(
    page_title="🔍 Advanced AI Search Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .search-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .result-box {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedSearchAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0.3,
            max_tokens=4096
        )
        self.setup_tools()
        self.setup_memory()
        self.setup_agent()
        
    def setup_tools(self):
        """Initialize all search tools"""
        # DuckDuckGo Search
        ddg_search = DuckDuckGoSearchAPIWrapper(max_results=10)
        
        # Wikipedia Search
        wikipedia = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
        
        # News Search Tool
        def news_search(query: str) -> str:
            """Search for recent news articles"""
            try:
                # Using DuckDuckGo news search
                search_query = f"{query} site:news.google.com OR site:reuters.com OR site:bbc.com"
                results = ddg_search.run(search_query)
                return f"📰 Latest News: {results}"
            except:
                return "News search temporarily unavailable"
        
        # Academic Search Tool
        def academic_search(query: str) -> str:
            """Search for academic papers and research"""
            try:
                search_query = f"{query} site:scholar.google.com OR site:arxiv.org OR site:pubmed.ncbi.nlm.nih.gov"
                results = ddg_search.run(search_query)
                return f"🎓 Academic Results: {results}"
            except:
                return "Academic search temporarily unavailable"
        
        # Image Search Tool
        def image_search(query: str) -> str:
            """Search for images related to the query"""
            try:
                search_query = f"{query} images"
                results = ddg_search.run(search_query)
                return f"🖼️ Image Search: {results}"
            except:
                return "Image search temporarily unavailable"
        
        # Video Search Tool
        def video_search(query: str) -> str:
            """Search for videos on the topic"""
            try:
                search_query = f"{query} site:youtube.com OR site:vimeo.com"
                results = ddg_search.run(search_query)
                return f"📹 Video Results: {results}"
            except:
                return "Video search temporarily unavailable"
        
        # Social Media Search
        def social_search(query: str) -> str:
            """Search social media platforms"""
            try:
                search_query = f"{query} site:twitter.com OR site:reddit.com OR site:linkedin.com"
                results = ddg_search.run(search_query)
                return f"📱 Social Media: {results}"
            except:
                return "Social media search temporarily unavailable"
        
        # Technical Documentation Search
        def tech_docs_search(query: str) -> str:
            """Search technical documentation"""
            try:
                search_query = f"{query} site:stackoverflow.com OR site:github.com OR site:docs.python.org"
                results = ddg_search.run(search_query)
                return f"💻 Tech Docs: {results}"
            except:
                return "Technical documentation search temporarily unavailable"
        
        # Shopping Search
        def shopping_search(query: str) -> str:
            """Search for products and shopping"""
            try:
                search_query = f"{query} site:amazon.com OR site:ebay.com OR price comparison"
                results = ddg_search.run(search_query)
                return f"🛒 Shopping: {results}"
            except:
                return "Shopping search temporarily unavailable"
        
        # Local Search
        def local_search(query: str) -> str:
            """Search for local businesses and services"""
            try:
                search_query = f"{query} near me OR local OR address OR phone"
                results = ddg_search.run(search_query)
                return f"📍 Local Results: {results}"
            except:
                return "Local search temporarily unavailable"
        
        self.tools = [
            Tool(
                name="General_Search",
                func=ddg_search.run,
                description="General web search for current information, facts, and various topics"
            ),
            Tool(
                name="Wikipedia_Search",
                func=wikipedia.run,
                description="Search Wikipedia for encyclopedic information, biographies, and historical facts"
            ),
            Tool(
                name="News_Search",
                func=news_search,
                description="Search for recent news articles and current events"
            ),
            Tool(
                name="Academic_Search",
                func=academic_search,
                description="Search for academic papers, research, and scholarly articles"
            ),
            Tool(
                name="Image_Search",
                func=image_search,
                description="Search for images related to the query"
            ),
            Tool(
                name="Video_Search",
                func=video_search,
                description="Search for videos and multimedia content"
            ),
            Tool(
                name="Social_Media_Search",
                func=social_search,
                description="Search social media platforms for discussions and opinions"
            ),
            Tool(
                name="Technical_Documentation",
                func=tech_docs_search,
                description="Search technical documentation, programming resources, and developer guides"
            ),
            Tool(
                name="Shopping_Search",
                func=shopping_search,
                description="Search for products, prices, and shopping information"
            ),
            Tool(
                name="Local_Search",
                func=local_search,
                description="Search for local businesses, services, and location-based information"
            )
        ]
    
    def setup_memory(self):
        """Setup conversation memory"""
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=10,
            return_messages=True
        )
    
    def setup_agent(self):
        """Initialize the search agent"""
        system_message = """You are an advanced AI search agent powered by Gemini 2.0 Flash. Your role is to:

1. Understand user queries and determine the best search strategy
2. Use appropriate tools based on the query type:
   - General_Search: For broad topics and current information
   - Wikipedia_Search: For encyclopedic and historical information
   - News_Search: For recent events and current news
   - Academic_Search: For research and scholarly content
   - Image_Search: For visual content queries
   - Video_Search: For multimedia and tutorial content
   - Social_Media_Search: For public opinions and discussions
   - Technical_Documentation: For programming and technical queries
   - Shopping_Search: For product and price comparisons
   - Local_Search: For location-based services

3. Synthesize information from multiple sources when needed
4. Provide comprehensive, accurate, and well-structured responses
5. Include source citations and timestamps when relevant
6. Identify potential biases or conflicting information

Always aim to provide the most helpful and accurate information possible."""

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate"
        )
    
    def search(self, query: str, search_type: str = "comprehensive") -> Dict[str, Any]:
        """Execute search with specified strategy"""
        try:
            # Modify query based on search type
            if search_type == "recent":
                enhanced_query = f"Recent information about {query} from 2024-2025"
            elif search_type == "comprehensive":
                enhanced_query = f"Comprehensive information about {query} including recent developments"
            elif search_type == "academic":
                enhanced_query = f"Academic research and studies about {query}"
            elif search_type == "news":
                enhanced_query = f"Latest news and current events about {query}"
            else:
                enhanced_query = query
            
            # Get response from agent
            response = self.agent.run(enhanced_query)
            
            return {
                "success": True,
                "response": response,
                "timestamp": datetime.datetime.now(),
                "query": enhanced_query
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.datetime.now(),
                "query": query
            }

def initialize_session_state():
    """Initialize session state variables"""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'search_analytics' not in st.session_state:
        st.session_state.search_analytics = {
            'total_searches': 0,
            'search_types': {},
            'response_times': []
        }

def display_search_analytics():
    """Display search analytics and metrics"""
    analytics = st.session_state.search_analytics
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Searches",
            analytics['total_searches'],
            delta=1 if analytics['total_searches'] > 0 else None
        )
    
    with col2:
        avg_time = sum(analytics['response_times']) / len(analytics['response_times']) if analytics['response_times'] else 0
        st.metric(
            "Avg Response Time",
            f"{avg_time:.2f}s",
            delta=f"{analytics['response_times'][-1]:.2f}s" if analytics['response_times'] else None
        )
    
    with col3:
        most_used = max(analytics['search_types'].items(), key=lambda x: x[1]) if analytics['search_types'] else ("None", 0)
        st.metric(
            "Most Used Type",
            most_used[0],
            delta=most_used[1]
        )
    
    with col4:
        success_rate = 95  # Placeholder - you'd calculate this based on actual success/failure rates
        st.metric("Success Rate", f"{success_rate}%")

def main():
    initialize_session_state()
    
    # Main title
    st.markdown('<h1 class="main-header">🔍 Advanced AI Search Agent</h1>', unsafe_allow_html=True)
    st.markdown("*Powered by LangChain + Gemini 2.0 Flash*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Enter your Google API key for Gemini 2.0 Flash"
        )
        
        if api_key:
            if st.session_state.agent is None:
                try:
                    st.session_state.agent = AdvancedSearchAgent(api_key)
                    st.success("✅ Agent initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Error initializing agent: {str(e)}")
        
        st.divider()
        
        # Search Configuration
        st.subheader("🎯 Search Settings")
        
        search_type = st.selectbox(
            "Search Strategy",
            ["comprehensive", "recent", "academic", "news", "technical"],
            help="Choose the type of search strategy"
        )
        
        max_results = st.slider("Max Results per Tool", 3, 20, 10)
        
        enable_multimodal = st.checkbox("Enable Multimodal Search", value=True)
        enable_fact_check = st.checkbox("Enable Fact Checking", value=True)
        enable_sentiment = st.checkbox("Sentiment Analysis", value=False)
        
        st.divider()
        
        # Advanced Features
        st.subheader("🚀 Advanced Features")
        
        search_filters = st.multiselect(
            "Content Filters",
            ["Remove Duplicates", "Academic Only", "Recent (Last 30 days)", "High Authority Sources"],
            default=["Remove Duplicates"]
        )
        
        export_format = st.selectbox(
            "Export Format",
            ["JSON", "CSV", "Markdown", "PDF"]
        )
        
        st.divider()
        
        # Search History
        st.subheader("📚 Search History")
        if st.session_state.search_history:
            for i, search in enumerate(reversed(st.session_state.search_history[-5:])):
                with st.expander(f"🔍 {search['query'][:30]}..."):
                    st.write(f"**Type:** {search.get('type', 'N/A')}")
                    st.write(f"**Time:** {search['timestamp'].strftime('%H:%M:%S')}")
                    if st.button(f"Re-run", key=f"rerun_{i}"):
                        st.session_state.current_query = search['query']
                        st.rerun()
        
        if st.button("🗑️ Clear History"):
            st.session_state.search_history = []
            st.session_state.search_analytics = {
                'total_searches': 0,
                'search_types': {},
                'response_times': []
            }
            st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Search interface
        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        st.subheader("🔍 Smart Search Interface")
        
        # Initialize query from session state if available
        if 'selected_template' in st.session_state:
            default_query = st.session_state.selected_template
            del st.session_state.selected_template
        else:
            default_query = ""
        
        # Query input with suggestions
        query_input = st.text_area(
            "Enter your search query:",
            value=default_query,
            height=100,
            placeholder="Ask me anything... I can search the web, academic papers, news, images, and more!",
            key="main_query"
        )
        
        # Search suggestions
        if query_input:
            suggestions = [
                f"Latest news about {query_input}",
                f"Academic research on {query_input}",
                f"How-to guides for {query_input}",
                f"Images related to {query_input}",
                f"Expert opinions on {query_input}"
            ]
            
            suggested_query = st.selectbox(
                "💡 Suggested Queries:",
                [""] + suggestions,
                key="suggestions"
            )
            
            if suggested_query:
                query_input = suggested_query
        
        # Search buttons
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        
        with col_btn1:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        with col_btn2:
            quick_search = st.button("⚡ Quick Search", use_container_width=True)
        
        with col_btn3:
            deep_search = st.button("🕳️ Deep Search", use_container_width=True)
        
        with col_btn4:
            multi_search = st.button("🌐 Multi-Source", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Execute search
        if (search_button or quick_search or deep_search or multi_search) and query_input and st.session_state.agent:
            search_start_time = datetime.datetime.now()
            
            # Determine search type based on button clicked
            if quick_search:
                current_search_type = "recent"
            elif deep_search:
                current_search_type = "comprehensive"
            elif multi_search:
                current_search_type = "academic"
            else:
                current_search_type = search_type
            
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Execute search
            with st.spinner("🤖 AI Agent is searching..."):
                status_text.text("Initializing search...")
                progress_bar.progress(20)
                
                status_text.text("Querying multiple sources...")
                progress_bar.progress(50)
                
                # Perform the search
                result = st.session_state.agent.search(query_input, current_search_type)
                progress_bar.progress(80)
                
                status_text.text("Processing results...")
                progress_bar.progress(100)
                
                # Calculate response time
                response_time = (datetime.datetime.now() - search_start_time).total_seconds()
                
                # Update analytics
                st.session_state.search_analytics['total_searches'] += 1
                st.session_state.search_analytics['response_times'].append(response_time)
                if current_search_type in st.session_state.search_analytics['search_types']:
                    st.session_state.search_analytics['search_types'][current_search_type] += 1
                else:
                    st.session_state.search_analytics['search_types'][current_search_type] = 1
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
            
            # Display results
            if result['success']:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.subheader(f"🎯 Search Results for: *{query_input}*")
                st.markdown(result['response'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add to search history
                st.session_state.search_history.append({
                    'query': query_input,
                    'type': current_search_type,
                    'response': result['response'],
                    'timestamp': result['timestamp'],
                    'response_time': response_time
                })
                
                # Export options
                st.subheader("📥 Export Results")
                col_export1, col_export2, col_export3 = st.columns(3)
                
                with col_export1:
                    if st.button("📋 Copy to Clipboard"):
                        st.code(result['response'])
                
                with col_export2:
                    export_data = {
                        'query': query_input,
                        'response': result['response'],
                        'timestamp': result['timestamp'].isoformat(),
                        'search_type': current_search_type
                    }
                    st.download_button(
                        "💾 Download JSON",
                        json.dumps(export_data, indent=2),
                        file_name=f"search_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col_export3:
                    markdown_content = f"""# Search Results
                    
**Query:** {query_input}
**Type:** {current_search_type}
**Timestamp:** {result['timestamp']}

## Response
{result['response']}
"""
                    st.download_button(
                        "📝 Download Markdown",
                        markdown_content,
                        file_name=f"search_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
            else:
                st.error(f"❌ Search failed: {result['error']}")
    
    with col2:
        # Analytics and insights
        st.subheader("📊 Search Analytics")
        display_search_analytics()
        
        # Search type distribution
        if st.session_state.search_analytics['search_types']:
            st.subheader("🎯 Search Type Distribution")
            df_types = pd.DataFrame(
                list(st.session_state.search_analytics['search_types'].items()),
                columns=['Type', 'Count']
            )
            fig = px.pie(df_types, values='Count', names='Type', title="Search Types Used")
            st.plotly_chart(fig, use_container_width=True)
        
        # Response time trend
        if len(st.session_state.search_analytics['response_times']) > 1:
            st.subheader("⏱️ Response Time Trend")
            times_df = pd.DataFrame({
                'Search': range(1, len(st.session_state.search_analytics['response_times']) + 1),
                'Response Time (s)': st.session_state.search_analytics['response_times']
            })
            fig = px.line(times_df, x='Search', y='Response Time (s)', title="Response Time Over Searches")
            st.plotly_chart(fig, use_container_width=True)
        
        # Quick search templates
        st.subheader("🚀 Quick Search Templates")
        
        templates = {
            "📰 Breaking News": "latest breaking news today",
            "🧬 Scientific Discoveries": "recent scientific breakthroughs 2024",
            "💼 Market Trends": "current market trends and analysis",
            "🌍 World Events": "major world events this week",
            "💻 Tech Updates": "latest technology updates and releases",
            "📈 Economic News": "economic indicators and financial news",
            "🎯 Industry Analysis": "industry analysis and insights",
            "🔬 Research Papers": "recent research papers and studies"
        }
        
        for template_name, template_query in templates.items():
            if st.button(template_name, use_container_width=True):
                st.session_state.selected_template = template_query
                st.rerun()
    
    # Advanced features section
    st.divider()
    
    # Multi-query search
    st.subheader("🔄 Multi-Query Search")
    st.write("Search multiple related queries simultaneously")
    
    col_multi1, col_multi2 = st.columns(2)
    
    with col_multi1:
        multi_queries = st.text_area(
            "Enter queries (one per line):",
            height=100,
            placeholder="Query 1\nQuery 2\nQuery 3"
        )
    
    with col_multi2:
        if st.button("🚀 Execute Multi-Search") and multi_queries and st.session_state.agent:
            queries = [q.strip() for q in multi_queries.split('\n') if q.strip()]
            
            for i, query in enumerate(queries):
                st.write(f"**Query {i+1}:** {query}")
                result = st.session_state.agent.search(query, search_type)
                if result['success']:
                    st.success(result['response'][:200] + "...")
                else:
                    st.error(f"Failed: {result['error']}")
                st.divider()
    
    # Search comparison
    st.subheader("⚖️ Search Comparison")
    st.write("Compare results from different search strategies")
    
    if st.session_state.search_history:
        comparison_query = st.selectbox(
            "Select a previous query to compare:",
            [h['query'] for h in st.session_state.search_history[-10:]]
        )
        
        if st.button("🔄 Compare Search Strategies") and comparison_query:
            strategies = ["comprehensive", "recent", "academic", "news"]
            
            for strategy in strategies:
                with st.expander(f"📊 {strategy.title()} Search Results"):
                    result = st.session_state.agent.search(comparison_query, strategy)
                    if result['success']:
                        st.write(result['response'])
                    else:
                        st.error(f"Error: {result['error']}")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Advanced Search Agent | Powered by LangChain & Gemini 2.0 Flash</p>
        <p>Features: Multi-source search, Real-time analysis, Export capabilities, Search analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()