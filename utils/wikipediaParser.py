from bs4 import BeautifulSoup
import re
import os
import pandas as pd
from tqdm import tqdm
from neo4j import GraphDatabase
import Levenshtein

reset_diz= {
    # Lists all pages nodes
    "Page.csv": "page_uri",
    # Lists all subChapters nodes
    "subChapter.csv": "subC_uri",
    # Lists all subChapters nodes
    "Chunk.csv": "chunk_uri,chunk_content",
    # Connect chunk to pages (cite relationship)
    "cites.csv": "chunk_uri,page_uri",
    # Connect page/subC to subC (has_subchapter relationship)
    "has_subchapter.csv": "uri,rel_type,subC_uri",
    # Connect subC to page (mainPage relationship)
    "mainpage.csv": "chapter_uri,main_uri",
    # Connect page/subC to chunk (has_chunk relationship)
    "has_chunk.csv": "uri,chunk_uri",
    # Connect chunk to chunk (next/previous relationship)
    "next.csv": "prev_chunk,next_chunk",
    # Connect pages to pages (DBrelation)
    "DBrelations.csv": "a,relType,b",
    # Add label to pages (DBlabels)
    "DBlabels.csv": "a,label",
    # Annotate questions
    "answers.csv": "chunk_id,question"
}


# Deletes all the data in all of the files in the 'csv' folder
def resetCsv():
    for csv_name in os.listdir("csv"):
        if csv_name == ".DS_Store":
            continue
        f = open("csv/" + csv_name, "w")
        f.write(reset_diz[csv_name])
        f.close()


# Computes the similarity between two strings
def similarity_score(a, b):
    return Levenshtein.ratio(a, b)


# Find Main article
def extractMain(tag):
    for next_sibling in tag.parent.find_next_siblings():
        if next_sibling.name[0] == "h":
            return
        if "Main article" in next_sibling.text:
            tag = next_sibling.find("a")
            return tag.get('title', '')
        

def formatName(str):
    str = re.sub('"', '""', str)
    str = re.sub("\[[0-9]{1,3}\]","", str)
    return str


def removeLastLine(file:str):
    # Code from "https://stackoverflow.com/questions/1877999/delete-final-line-in-file-with-python"
    
    with open("csv/" + file, "rb+") as file:
        file.seek(0, os.SEEK_END)
        pos = file.tell() - 1
        while pos > 0 and file.read(1) != b'\n':
            pos -= 1
            file.seek(pos, os.SEEK_SET)
        if pos > 0:
            file.seek(pos, os.SEEK_SET)
            file.truncate()


def chunkExtractor(parent_uri, tag_list):
    i = 0
    previous_name = ""
    # extract external links from subchapter
    for next_sibling in tag_list:
        if not next_sibling.name:
            continue
        # If the next element is a title, break
        if next_sibling.name[0] == "h":
            break
        # Cover the case in which a table is nested inside a div
        if next_sibling.name == "div" and next_sibling.find_all("table"):
            next_sibling = next_sibling.find_all("table")[0]
        # Parse paragraphs
        if next_sibling.name in ["p", "table", "dd", "dl"]:
            # If no character are present in the content of the chunk, skip it
            if not re.search('[a-zA-Z]', next_sibling.text):
                continue
            content = formatName(next_sibling.text.strip())
            # if the element is a table, raplace newlines characters with |
            if next_sibling.name == "table":
                content = re.sub(r"\n+", "|",content)
            # Clean content
            #content = re.sub(r'\[\d+\]', '', content)
            content = re.sub('{.*?} ', ' ', content.replace('\n', ''))
            content = content.encode('ascii',errors='ignore').decode('ascii')
            # If the content ends with double backslash it will break the csv format
            if content[-1] == "\\":
                content = content[:-1]
            # Define chunk_name
            i += 1
            chunk_name = parent_uri + "#" + str(i)
            # Write on Chunk.csv
            line = f'\n"{chunk_name}","{content}"'
            with open('csv/Chunk.csv', 'a') as chunkfile:
                chunkfile.write(line)
            # Write relationship on has_chunk.csv
            line = f'\n"{parent_uri}","{chunk_name}"'
            with open('csv/has_chunk.csv', 'a') as haschunkfile:
                haschunkfile.write(line)
            # Extract Links and write cites relationships
            cites(next_sibling.find_all("a"), chunk_name)
            # Write "next relationship"
            if previous_name:
                nextChunk(previous_name, chunk_name)
            previous_name = chunk_name
        # Parse ordered and unordered lists
        if next_sibling.name in ["ul", "ol", "div", "tr"]:
            # If we have a list as the first element of a subchapter, we are going to call it "List"
            if not previous_name:
                previous_name = chunk_name = parent_uri + "#1:List"
                content = ""
                # Write relationship on has_chunk.csv
                line = f'\n"{parent_uri}","{previous_name}"'
                with open('csv/has_chunk.csv', 'a') as haschunkfile:
                    haschunkfile.write(line)
            # Build the text from the list
            lst = []
            for li in next_sibling.find_all("li"):
                if not li:
                    continue
                lst.append(formatName(re.sub('{.*?} ', ' ', re.sub(r"\n+", "|",li.text))))
            lst = [x for x in lst if x != ""]
            # Order the list
            for x in range(len(lst) - 1):
                if lst[x + 1].strip() in lst[x]:
                    lst[x] = lst[x].split(lst[x+1])[0]
            if len(lst) == 0:
                continue
            text = ' + ' + ' + '.join(lst)
            # Remove last line if list is inside a chunk
            if content:
                removeLastLine("Chunk.csv")
            # Write chunk in Chunk.csv
            content += text
            if content[-1] == "\\":
                content = content[:-1]
            content = content.encode('ascii',errors='ignore').decode('ascii')
            line = f'\n"{previous_name}","{content}"'
            with open('csv/Chunk.csv', 'a') as chunkfile:
                chunkfile.write(line)
            # Extract Links and write cites relationships
            cites(next_sibling.find_all("a"), chunk_name)


def nextChunk(previous_name, chunk_name):
    line = f'\n"{previous_name}","{chunk_name}"'
    with open('csv/next.csv', 'a') as citefile:
        citefile.write(line)


def cites(a_lst, chunk_name):
    if not a_lst:
        return
    str_cites = ""
    for a in a_lst:
        if not a.has_attr("title"):
            continue
        link_title = formatName(a['title']).split('#')[0]
        link_title = link_title.replace(" (page does not exist)", "")
        if a.has_attr("href"):
            if "File:" in a["href"]:
                continue
            # Write new pages found in the links
            if not '#' in a['href']:
                writePage(link_title)
        line = f'\n"{chunk_name}","{link_title}"'
        str_cites += line
    # Write relationship on cites.csv
    with open('csv/cites.csv', 'a') as citefile:
        citefile.write(str_cites)


def isSubchapterOf(uri, rel_type, page_uri):
    str = f'\n"{uri}"'
    with open('csv/subChapter.csv', 'a') as subcfile:
        subcfile.write(str)
    str = f'\n"{page_uri}","{rel_type}","{uri}"'
    with open('csv/has_subchapter.csv', 'a') as subcfile:
        subcfile.write(str)


def mainPage(uri, main_uri):
    main_uri = formatName(main_uri.split('#')[0])
    main_uri = main_uri.replace(" (page does not exist)", "")
    if main_uri == "":
        return
    writePage(main_uri)
    str = f'\n"{uri}","{main_uri}"'
    with open('csv/mainpage.csv', 'a') as mainpfile:
        mainpfile.write(str)


def writePage(page_title):
    str = f'\n"{page_title}"'
    if str != ",":
        with open('csv/Page.csv', 'a') as pagefile:
            pagefile.write(str)


def removeTags(str):
    return re.sub("<i>|</i>|<span>|</span>", "", str)


def parse(page_obj):
    page_title = str(page_obj["document_title"])
    page_html = page_obj["document_html"]
    page_title = formatName(page_title)
    writePage(page_title)
    # initialize header_diz
    header_diz = {1: page_title}
    # Convert to BS object
    bsData = BeautifulSoup(page_html, "html.parser")
    # Find first header in the page
    tag = bsData.find('span', class_="mw-headline")
    # If the page does not have subtitles, extract all the chunks and continue
    if not tag:
        body = bsData.find('div', class_="mw-parser-output")
        chunkExtractor(page_title, body)
        return
    # Get all previous tags
    previous_tags = tag.parent.find_previous_siblings()
    # Reverse the list so that they are ordered correctly
    previous_tags.reverse()
    chunkExtractor(page_title, previous_tags)
    span_lst = bsData.find_all('span', class_="mw-headline")
    #sections = [span.text for span in span_lst]
    for span in span_lst:
        section_name = span.get('id', span.text)
        section_name = section_name.replace("_", " ")
        if section_name == "References" or section_name == "See also":
            break
        if span.parent.name[0] != "h":
            continue
        # Get section level
        section_name = removeTags(section_name)
        level = int(span.parent.name[1])
        # create uri
        uri = header_diz[level - 1] + "#" + formatName(section_name)
        # Write uri in header_diz
        header_diz[level] =  uri
        # Write subchapter relation
        isSubchapterOf(uri, formatName(section_name).lower(), header_diz[level - 1])
        # Search for Mainpage
        main_uri = extractMain(span)
        if main_uri:
            mainPage(uri, main_uri)
        # Search for links
        chunkExtractor(uri, span.parent.find_next_siblings())


def createConstrains():
    file = open("db commands.txt", "r")
    content = file.read().split("\n\n")
    file.close
    URI = "neo4j://localhost"
    AUTH = ("neo4j", "Giardo2000")
    for chunk in content[:4]:
        comment = chunk.split('\n')[0].replace('//', '').strip()
        print(comment)
        command = '\n'.join(chunk.split('\n')[1:])
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            session = driver.session()
            session.run(command)
    # Check if all constrains have properly been applied
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        index_list, summary, keys = driver.execute_query(
            "SHOW INDEXES;",
        )
    index_list = [x['name'] for x in index_list]
    if not all(e in index_list for e in ['chunk_uri', 'page_uri', 'resource_uri', 'subchapter_uri']):
        createConstrains()


def neo4jExecutor():
    file = open("db commands.txt", "r")
    content = file.read().split("\n\n")
    file.close
    URI = "neo4j://localhost"
    AUTH = ("neo4j", "Giardo2000")
    for chunk in content[4:]:
        comment = chunk.split('\n')[0].replace('//', '').strip()
        print(comment)
        command = '\n'.join(chunk.split('\n')[1:])
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            session = driver.session()
            session.run(command)
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        uri_lst, summary, keys = driver.execute_query(
            "MATCH (n:Page) RETURN n.uri;",
        )
    query_lst = ""
    for uri in uri_lst:
        query_lst += '"http://en.wikipedia.org/wiki/'
        query_lst += uri['n.uri'].replace(" ", "_").replace('"','\\"')
        query_lst += '",'
    query_lst = query_lst[:-1]
    return query_lst


def linkChunks():
    URI = "neo4j://localhost"
    AUTH = ("neo4j", "Giardo2000")
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        id_lst, summary, keys = driver.execute_query(
            "MATCH (n:OriginalPage) RETURN ID(n);",
        )
    for page_id in tqdm(id_lst):
        id_lst, summary, keys = driver.execute_query(
            """MATCH (p:OriginalPage)-[:has_chunk]->(c1:Chunk)
            MATCH (p)-[:has_chunk]->(c2:Chunk)
            WHERE id(p) = $page_id AND id(c1) < id(c2) AND NOT (c1)-[:next]->(c2)
            CREATE (c1)-[:same_page]->(c2), (c1)<-[:same_page]-(c2);""",
            page_id = page_id['ID(n)']
        )


def extractRelations(query_lst):
    URI = "neo4j://localhost"
    AUTH = ("neo4j", "Giardo2000")
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        rel_lst, summary, keys = driver.execute_query(
            """WITH [%s] AS list
            MATCH (a:WikipediaPage)<-[:isPrimaryTopicOf]-()-[r]->()-[:isPrimaryTopicOf]->(b:WikipediaPage)
            WHERE a.uri IN list AND b.uri IN list AND NOT a = b
            RETURN a.uri,TYPE(r),b.uri""" % query_lst,
        )
    print("found " + str(len(rel_lst)) + " DBpedia relationships")
    for rel in rel_lst:
        writeRelation(rel['a.uri'][29:].replace("_", " "),rel['TYPE(r)'],rel['b.uri'][29:].replace("_", " "))


def extractLabels(query_lst):
    URI = "neo4j://localhost"
    AUTH = ("neo4j", "Giardo2000")
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        label_lst, summary, keys = driver.execute_query(
            """WITH [%s] AS list
            MATCH (a:WikipediaPage)<-[:isPrimaryTopicOf]-(b)
            WHERE a.uri IN list
            RETURN a.uri,labels(b)""" % query_lst,
        )
    print("found " + str(len(label_lst)) + " DBpedia labels")
    for resource in label_lst:
        for label in resource['labels(b)']:
            if label in ["Resource", "Thing"]:
                continue
            writeLabel(resource['a.uri'][29:].replace("_", " "),label)


def writeRelation(a,rel,b):
    str = f'\n"{a}","{rel}","{b}"'
    with open('csv/DBrelations.csv', 'a') as relfile:
        relfile.write(str)


def writeLabel(a,label):
    str = f'\n"{a}","{label}"'
    with open('csv/DBlabels.csv', 'a') as labelfile:
        labelfile.write(str)


def resetDb():
    URI = "neo4j://localhost"
    AUTH = ("neo4j", "Giardo2000")
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        session = driver.session()
        session.run(
            """MATCH (n:Page)
            CALL { WITH n
            DETACH DELETE n
            } IN TRANSACTIONS OF 1000 ROWS;"""
        )
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        session = driver.session()
        session.run(
            """MATCH (n:SubChapter)
            CALL { WITH n
            DETACH DELETE n
            } IN TRANSACTIONS OF 1000 ROWS;"""
        )
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        session = driver.session()
        session.run(
            """MATCH (n:Chunk)
            CALL { WITH n
            DETACH DELETE n
            } IN TRANSACTIONS OF 1000 ROWS;"""
        )
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        rel_lst, summary, keys = driver.execute_query(
            "CALL apoc.schema.assert({}, {})"
        )


def addDBpediaInfo():
    URI = "neo4j://localhost"
    AUTH = ("neo4j", "Giardo2000")
    print("Adding DBpedia relationships")
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        session = driver.session()
        session.run(
            """LOAD CSV WITH HEADERS FROM 'file:///DBrelations.csv' AS row
            CALL {WITH row
            MATCH (node0:Page {uri: row.a})
            MATCH (node1:Page {uri: row.b})
            WITH node0, node1, row, 'is ' + row.relType + ' of' AS new_rel_type
            CREATE (node0)-[:vector_rel {source: "DBpedia", text: row.relType}]->(node1), (node0)<-[:vector_rel {source: "DBpedia", text: new_rel_type}]-(node1)} IN TRANSACTIONS OF 5000 ROWS;"""
        )
    print("Adding DBpedia labels")
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        session = driver.session()
        session.run(
            """LOAD CSV WITH HEADERS FROM 'file:///DBlabels.csv' AS row
            CALL {WITH row
            MATCH (node0:Page {uri: row.a})
            WITH node0, node0.text + ' (' + row.label + ')' AS new_text
            SET node0.text = new_text} IN TRANSACTIONS OF 5000 ROWS;"""
        )


def extractAnswers(jsonObj, URI, AUTH):
    doc_title = jsonObj["document_title"][0]
    question = jsonObj["question_text"][0]
    # Create answers list
    answers = []
    for ann in jsonObj['annotations'][0]:
        if ann['long_answer']['start_byte'] == -1:
            continue
        answer = [x['token'] for x in jsonObj['document_tokens'][0][ann['long_answer']['start_token']:ann['long_answer']['end_token']]]
        if answer in answers:
            continue
        answers.append(answer)
    # If no answer, skip
    if len(answers) == 0:
        return question, "No answer provided"
    # Extract from Neo4j all the chunks of the page
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        chunk_lst, summary, keys = driver.execute_query(
            "MATCH (p:Page {uri: $title})-[:has_chunk]->(c:Chunk) RETURN DISTINCT ID(c), c.text;",
            title = doc_title
        )
    # If the page has not been found in Neo4j, skip
    if len(chunk_lst) == 0:
        print("Couldn't find " + str(doc_title))
        return question, "Page not found in Neo4j"
    # For each answer and each chunk, check if chunk contains the answer
    ans_chunk = []
    found = False
    for answer in answers:
        answer = [x for x in answer if re.findall("^[a-zA-Z]+$", x)]
        for chunk in chunk_lst:
            z = 0
            for token in answer:
                if token.lower() in chunk['c.text'].lower():
                    z += 1
            # If it does, write it in answers.csv
            if z/len(answer) > 0.8 and chunk['ID(c)'] not in ans_chunk:
                chunk_id = chunk['ID(c)']
                ans_chunk.append(chunk_id)
                line = f'\n"{chunk_id}","{question}"'
                with open('csv/answers.csv', 'a') as answerfile:
                    answerfile.write(line)
                    found = True
    if found:
        return question, "found"
    else:
        return question, "not found"


def writeAnswers(URI, AUTH):
    # Extract from Neo4j all the chunks of the page
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        session = driver.session()
        session.run("""LOAD CSV WITH HEADERS FROM 'file:///answers.csv' AS row
            CALL {WITH row
            MATCH (chunk:Chunk) WHERE ID(chunk) = toInteger(row.chunk_id)
            SET chunk.is_answer_of = row.question} IN TRANSACTIONS OF 10000 ROWS;"""
        )