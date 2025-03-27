import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Query
from textwrap import dedent
from typing import Optional

from fastapi.security.api_key import APIKeyHeader
from crewai import Crew,Agent, Task, Process, LLM
from crewai.tools import BaseTool
import json
from langchain_community.utilities import SQLDatabase
from pydantic import Field

from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDatabaseTool,
)

load_dotenv("prod.env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
API_KEY = os.getenv("ACQ_API_KEY")
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


async def verify_api_key(api_key: str = Depends(api_key_header)):
    print(f"Received API Key: {api_key}")
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

app = FastAPI(title="Server for SR")

llm=LLM(model='gemini/gemini-2.0-flash',api_key=GOOGLE_API_KEY)
DATABASE_URI = "postgresql://postgres:admin@localhost:5432/final"
db = SQLDatabase.from_uri(DATABASE_URI)



class ListTablesTool(BaseTool):
    name: str = Field(default="list_tables", description="List all tables in the PostgreSQL database")
    description: str = Field(default="List all tables in the database.")

    def _run(self) -> str:
        return ListSQLDatabaseTool(db=db).invoke("")

class ExecuteSQLTool(BaseTool):
    name: str = Field(default="execute_sql", description="Execute a SQL query against the database.")
    description: str = Field(default="Execute a SQL query and return the result.")

    def _run(self, sql_query: str) -> str:
        return QuerySQLDatabaseTool(db=db).invoke(sql_query)

class TablesSchemaTool(BaseTool):
    name: str = Field(default="tables_schema", description="Retrieve table schema and sample rows.")
    description: str = Field(default="Get schema and sample rows for given tables.")

    # def _run(self, tables: str) -> str:
    #     tool = InfoSQLDatabaseTool(db=db)
    #     return tool.invoke(tables)

    def _run(self, tables: Optional[str] = None) -> str:
        if tables is None:
            raise Exception("Tables parameter is required for retrieving schema.")
        
        if isinstance(tables, list):
            tables = ', '.join(tables)  # Convert list to comma-separated string
        elif not isinstance(tables, str):
            raise Exception(f"Invalid input type for tables: {type(tables)}. Expected string or list.")
            
        tool = InfoSQLDatabaseTool(db=db)
        return tool.invoke(tables)


# class CheckSQLTool(BaseTool):
#     name: str = Field(default="check_sql", description="Check if SQL query is valid.")
#     description: str = Field(default="Validate SQL query before execution.")
#     def _run(self, sql_query: str) -> str:
#         return QuerySQLCheckerTool(db=db, llm=llm).invoke({"query": sql_query})


list_tables_tool = ListTablesTool()
execute_sql_tool = ExecuteSQLTool()
tables_schema_tool = TablesSchemaTool()



gdp_agent=Agent(

    role="gdp_database_engineer",
    goal="Construct sql queries and execute SQL queries based on a request",
    backstory="""You are an experienced database engineer who is master at creating efficient and complex SQL queries.
        You have a deep understanding of how different databases work and how to optimize queries.
        Use the `list_tables` to find available tables.
        Use the `tables_schema` to understand the metadata for the tables.
        Use the `execute_sql` to check your queries for correctness.
        Use the following below tables to get the data and ensure to pass table names as a single string, not a list, when using the tables_schema_tool:
                        a.annual_estimate_gdp_crore
                        b.annual_estimate_gdp_growth_rate
                        c.gross_state_value
                        d.key_aggregates_of_national_accounts
                        e.per_capita_income_product_final_consumption
                        f.provisional_estimateso_gdp_macro_economic_aggregates
                        g.quaterly_estimates_of_expenditure_components_gdp
                        h.quaterly_estimates_of_gdp
                        If table has current and constant prices, return results for both.
        You use data and finally provide json results based on sql query executed.
       You always include all the fields details so that other person will know what this data is about.
       And always give exact numerical values .

       Ensure that:
    1. All outputs must be returned in a clean, valid JSON format.
    2. Avoid using unnecessary text or explanations outside of the JSON.
    3. Follow the correct JSON structure without any syntax issues (e.g., no trailing commas, double quotes around keys and values).
    4. If a query fails or has no results, return a json response in this format:
       ["status = Your Error description here"]
    5. Ensure JSON formatting is strict, following RFC 8259 JSON standards.
    6. Provide accurate results using the tables specified.
    
    """,
    llm=llm,
    tools=[list_tables_tool, tables_schema_tool,execute_sql_tool],
    allow_delegation=False,
    verbose=True,
    )

gdp_extract_data=Task(

    description="""Extract the data that is required for the  user query:{query}.
                   Make sure numerical values are always actual and correct as mentioned in the database.
                   Before executing the query make sure column names are always correct and make use of all the available tools.
                   Make sure information asked for year exists if not exists then return a  response in this format:["status = Your Error description here"]
                   If no data is aviilable related to user query then inform user or ask for more context.

                        Use the following below tables only to get the data:
                        a.annual_estimate_gdp_crore
                        b.annual_estimate_gdp_growth_rate
                        c.gross_state_value
                        d.key_aggregates_of_national_accounts
                        e.per_capita_income_product_final_consumption
                        f.provisional_estimateso_gdp_macro_economic_aggregates
                        g.quaterly_estimates_of_expenditure_components_gdp
                        h.quaterly_estimates_of_gdp
                        If table has current and constant prices, return results for both.


Important!!!=>When setting conditions for column values, always refer to the following list for each table. This list contains the exact string matches to ensure SQL query conditions are accurate. By following this, you can avoid any mistakes while generating SQL queries
a.quaterly_estimates_of_expenditure_components_gdp=["Private final consumption expenditure","Government final consumption expenditure" , "Gross fixed capital formation","Change in stock","Valuables","Exports of goods and services","Less: Imports of goods and services","Discrepancies","Gross domestic product","Current Price","Constant Price","Q1","Q2","Q4","Q3"]
b.quaterly_estimates_of_gdp=["Primary Sector","Agriculture, Livestock, Forestry & Fishing","Mining & Quarrying","Secondary Sector","Manufacturing","Electricity, Gas, Water Supply & Other Utility Services","Construction","Tertiary Sector","Trade, Hotels, Transport, Communication & Services related to Broadcasting","Financial, Real Estate & Professional Services","Public Administration, Defence & Other Services","GVA at Basic Prices","GDP ","GFCE","PFCE","GFCF","CIS","Valuables","Exports of goods and services","Imports of goods and services","Discrepancies* ","constant price","current price","Q3","Q1","Q2","Q4"]
c.provisional_estimateso_gdp_macro_economic_aggregates=["Gross National Income (GNI)","Net National Income (NNI)","Gross Value Added (GVA) at basic prices","Gross Domestic Product (GDP)","Net Domestic Product (NDP)","Private final consumption expenditure","Government final consumption expenditure","Gross fixed capital formation","Change in stocks","Valuables","Exports","Less Imports","Discrepancies","Per capita NNI","Current Prices","Constant Prices","Domestic Product","ESTIMATES AT PER CAPITA LEVEL","National Income"]
d.per_capita_income_product_final_consumption=["Per Capita GDP( ₹) ","Per Capita GNI ( ₹) ","Per Capita NNI ( ₹) ","Per Capita GNDI( ₹) ","Per Capita PFCE( ₹) ","current price","constant price","Percentage change over previous year at constant (2011-12) prices"]
e.key_aggregates_of_national_accounts=["GVA at basic prices","Taxes on Products including import duties","Less Subsidies on Products","CFC","NDP","PFCE","GFCE","GCF","GFCF","CIS","VALUABLES","Exports of goods and services","Export of goods","Export of services","Less Imports of goods and services","Import of goods","Import of services","Discrepancies","GDP","Primary income receivable from ROW (net)","GNI","NNI","Other current transfers (net) from ROW","GNDI","NNDI","Gross Saving","Net Saving","Gross Saving to GNDI","GCF to GDP","GCF  excluding Valuables to GDP","PFCE to NNI","current price","constant price","Domestic Product","Final Expenditure","Rates","Rates of Expenditure Components to GDP"]
f.gross_state_value=["Agriculture, forestry and fishing","Crops","Livestock","Forestry and logging","Fishing and aquaculture","Mining and quarrying","Primary","Manufacturing","Electricity, gas, water supply & other utility services","Construction","Secondary","Trade, repair, hotels and restaurants","Trade & repair services","Hotels & restaurants","Transport, storage, communication & services related to broadcasting","Railways","Road transport","Water transport",
                     "Air transport","Services incidental to transport","Storage","Communication & services related to broadcasting","Financial services","Real estate, ownership of dwelling & professional services","Public administration","Other services","Tertiary","TOTAL GSVA at basic prices","Taxes on Products","Subsidies on products","Gross State Domestic Product","Population","Per Capita GSDP","constant price","current price"
                     ,"Jammu & Kashmir","Chhatisgarh","Jharkhand","Uttarakhand" ,etc....]
e.annual_estimate_gdp_growth_rate and annual_estimate_gdp_crore=["Primary Sector","Agriculture, Livestock, Forestry & Fishing","Mining & Quarrying","Secondary Sector","Manufacturing","Electricity, Gas, Water Supply & Other Utility Services","Construction","Tertiary Sector","Trade, Hotels, Transport, Communication & Services related to Broadcasting","Financial, Real Estate & Professional Services","3.3 Public Administration, Defence & Other Services","GVA at Basic Prices","GVA at Basic Prices","NVA at Basic Prices","GNI ","NNI ","Per capita income(Rs.)","Net taxes on Products","GDP ","GFCE","PFCE","GFCF","CIS","Valuables","Exports of goods and services","Exports of goods and services","Discrepancies ","constant","current"]

                        
Information about the tables that you can use to spot the table as most of the tables have same item names but these table serves different purpose so identify the purpose using below information:
["annual_estimate_gdp_crore:Represents the GDP estimates (in crores) for each sector or sub-sector of the economy during the specified time period.",
"annual_estimate_gdp_growth_rate:Represents the growth rate percentage for each sector at the given prices.",
"gross_state_value: id: Represents the Gross State Value Added (GSVA) for each sector in crores for the year.",
"key_aggregates_of_national_accounts: id: here item: Refers to the economic metric or component being measured, value: Represents the value of the item in crores, price_type: Specifies the price basis for the data, value_category: Denotes the broad economic category the item belongs to.",
"per_capita_income_product_final_consumption: here item: Refers to the economic measure being reported on a per capita (per person) basis, population: Represents the population figure (in crores) used to calculate the per capita measures, value_category:value: Represents the per capita value (in ₹) for each item."
"provisional_estimateso_gdp_macro_economic_aggregates: here item: Represents the economic metric being measured, category: Specifies the broader economic category the item falls under, value: Represents the numerical value of each economic metric in crores of rupees (₹) for the year."
"quaterly_estimates_of_expenditure_components_gdp:item Refers to the specific component of GDP being measured, quarter Indicates the specific quarter for which the data is reported, value Refers to the total value of output (in ₹ crore) for each sector in the specified quarter, price_type:.",
"quaterly_estimates_of_gdp: here item Refers to the specific component of GDP being measured, quarter Indicates the specific quarter for which the data is reported, growth_value  Represents the monetary value (in ₹ crore) of each item during the respective quarter, value Refers to the total value of output (in ₹ crore) for each sector in the specified quarter"
  
]

a.Format of years columns are like this : 2023-2024.
b.so when asked query for year for example 2023 you take it like 2023-24.
c.Also make sure when error comes in years column you check the schema and few rows of the tables to know the fromat.
d.Dont give id column of the tables in the json response.
e.Always try navigate the unique values in the columns so that you choose the most relevant table and give coorect result.


Some rules that you should follow:
1. **ENSURE ALL SELECT STATEMENTS INCLUDE THE All THE RELEVANT  FIELDS:**  
   - Queries **must be fully structured and accurate**.  

2. **DEFAULT CONDITIONS (UNTIL SPECIFIED OTHERWISE IN THE QUESTION):**  
   - `WHERE year = MAX(year).  

   
### **Agent Responsibilities**
1. **SQL Query Generation:**
   - Generate only `SELECT` statements (No `UPDATE`, `DELETE`, or `INSERT` or `DROP` or `TRUNCATE`).
   - Construct queries based on user intent while ensuring accuracy.
   - Handle aggregation (`AVG`, `SUM`, `MAX`, `MIN`), filtering (`WHERE`),when needed use groupby(), and sorting (`ORDER BY`) based on context ,also use other sql functions (for example LAG(),STDDEV() etc) if needed.

2. **High-Level Validation:**
   - Ensure the generated SQL query is syntactically valid.
   - Verify column names and table existence before execution.
   - Prevent SQL injection by sanitizing input.
   - If the query is ambiguous, infer intent based on available metadata use tools for that.

### Rules for Query Generation:
1. Ensure queries are **optimized** and **structured correctly**.
2. If the user asks for trends, perform **aggregations** or **time-series comparisons** as needed.
3. If the user asks for a summary, **group data** by relevant fields.
4. If the user asks for state-wise comparisons, include **GROUP BY state**.
5. If the user asks for inflation-related queries, focus on **inflation_percentage**.
6. If the user requests data for a specific period, use **year**  filter.
  

   
    """,
    expected_output="Database result for the query in valid json",
    agent=gdp_agent,
    verbose=True,    )



crew = Crew(
    agents=[gdp_agent, ],
    tasks=[gdp_extract_data, ],
    verbose=True,)


import re
import json

def handle_json_response(raw_data):
    try:
        if not isinstance(raw_data, str):
            raise Exception("Invalid or missing raw data in the agent response.")
        
        # Clean up JSON if needed
        raw_data = raw_data.strip("```json").strip().strip("```")

        # Attempt direct parsing
        try:
            return json.loads(raw_data)
        except json.JSONDecodeError:
            pass
        
        # Fix common issues: Single quotes to double quotes, remove trailing commas
        cleaned_data = re.sub(r"'", '"', raw_data)
        cleaned_data = re.sub(r",\s*([}\]])", r"\1", cleaned_data)

        # Try parsing cleaned JSON
        try:
            return json.loads(cleaned_data)
        except json.JSONDecodeError as e:
            line, column = e.lineno, e.colno
            error_context = cleaned_data.splitlines()[line - 1] if line <= len(cleaned_data.splitlines()) else "Unknown"
            print(f"JSON Decode Error at Line {line}, Column {column}: {e}")
            print(f"Context: {error_context}")
            raise Exception(f"Try again Unable to fix JSON. Error at Line {line}, Column {column}: {e}")
    except Exception as e:
        raise Exception(f"Error processing JSON data: {str(e)}")

@app.get("/query_gdp" ,dependencies=[Depends(verify_api_key)])
async def run_query(user_query: str = Query(..., description="Natural language SQL query")):
    """Convert user query to SQL and execute it using an agent with table context."""
    try:
        inputs = {"query": user_query}
        result = crew.kickoff(inputs=inputs)
        print("Raw LLM Response:", result)  
       
        if not result  or not hasattr(result, 'tasks_output') or not result.tasks_output:
            raise Exception("No output found from agents.  check your query")

        first_output = result.tasks_output[0]
        raw_data = getattr(first_output, 'raw', None)

        if not raw_data or not isinstance(raw_data, str):
            raise Exception("Invalid or missing raw data in the agent response.")

        parsed_data = handle_json_response(raw_data)
        
        # Convert raw JSON string to Python object
        # try:
        #     parsed_data = json.loads(raw_data.strip("```json").strip().strip("```"))
        # except json.JSONDecodeError as e:
        #     raise Exception(f"Run again Error decoding JSON data: {str(e)}")

        print("Parsed Data:", parsed_data)

        structured_response = {
            "query": user_query,
            "result": parsed_data
        }
        return structured_response

    except HTTPException as e:
        raise e
    except Exception as e:
        error_message = str(e)
        print(f"Error: {error_message}")
        return {"error": error_message}






# class CheckSQLTool(BaseTool):
#     name: str = Field(default="check_sql", description="Check if SQL query is valid.")
#     description: str = Field(default="Validate SQL query before execution.")
#     def _run(self, sql_query: str) -> str:
#         return QuerySQLCheckerTool(db=db, llm=llm).invoke({"query": sql_query})

# check_sql_tool = CheckSQLTool()
