import os
import re
import json
from operator import itemgetter
import nltk
from flask import Flask, render_template, request, jsonify
import requests
from datetime import datetime
import time
import pandas as pd
from flask import send_file
import operator

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
data_path = os.getcwd()
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a more secure secret key

class SearchApp:
    def __init__(self):
        self.data_path = os.getcwd() + '/document_collection/'
        self.docID_map = {}
        self.load_documents()
        self.glfilename = ''
        self.gltitle = ''
    def load_documents(self):
        self.files = os.listdir(self.data_path)
        self.docID = 1
        for filename in self.files:
            self.docID_map[self.docID] = filename
            self.docID += 1

    def read_file(self, filename):
        file_path = os.path.join(self.data_path, filename)
        with open(file_path, encoding='latin-1') as file:
            lines = file.readlines()

            # Assuming the first line contains the headers
            headers = lines[0].strip().split('\t')
            title_index = headers.index('title') if 'title' in headers else -1
            overview_index = headers.index('overview') if 'overview' in headers else -1

            if title_index == -1 or overview_index == -1:
                print("Error: 'title' or 'overview' column not found in document.")
                return "", ""

            # Extracting the title and overview columns
            title_column = [line.strip().split('\t')[title_index] if len(line.strip().split('\t')) > title_index else ''
                            for line in lines[1:]]
            overview_column = [
                line.strip().split('\t')[overview_index] if len(line.strip().split('\t')) > overview_index else '' for
                line in lines[1:]]

            return title_column, overview_column

    def preprocess(self, text):
        normalized = []
        text = text.lower()
        text = re.sub(r'\b[0-9]+\b', '', text)
        tokens = re.findall(r'\b\w+\b', text)
        text_list = [i for i in tokens if i not in stop_words]
        for word in text_list:
            word = wordnet_lemmatizer.lemmatize(word)
            normalized.append(word)
        return normalized

    def poster(self, text):
        base_url = "http://www.omdbapi.com/"
        params = {
            't': text,
            'apikey': 'd1354d4c',
        }

        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code == 200 and data['Response'] == 'True':
            poster_url = data['Poster']
            return poster_url
        else:
            print(f"Error: {data['Error']}")
            api_key = "136a570380da6bccef7f8b3601632067"
            base_url = "https://api.themoviedb.org/3"
            search_query = text
            search_endpoint = f"{base_url}/search/movie"
            params = {"api_key": api_key, "query": search_query}

            response = requests.get(search_endpoint, params=params)
            data = response.json()
            if data["results"]:
                poster_path = data["results"][0]["poster_path"]
                full_poster_path = f"https://image.tmdb.org/t/p/original{poster_path}"
                return full_poster_path
            return None


    def create_index(self, terms, docID):
        with open(data_path+'/dict.json', 'r') as index:
            ivdict = json.loads(index.read())
        for term in terms:
            if term in ivdict:
                postings_list = [posting[0] for posting in ivdict[term]]
                if docID in postings_list:
                    ivdict[term][-1][1] += 1
                else:
                    ivdict[term].append([docID, 1])
            else:
                ivdict.update({term: [[docID, 1]]})
        index_file = open(data_path+"/dict.json", "w")
        index_file.write(json.dumps(ivdict))
        index_file.close()

    def search(self, query):
        query = self.preprocess(query)
        query_output = self.search_documents(query)

        results = []
        vote_average_column, runtime_column = [], []  # Initialize empty lists

        for doc_item in query_output:
            if doc_item in self.docID_map:
                filename = self.docID_map[doc_item]
                titles, overviews = self.read_file(filename)

                for title, overview in zip(titles, overviews):
                    if any(term in title.lower() for term in query):
                        postimg = self.poster(title)
                        result_entry = {
                            'title': title,
                            'overview': overview,
                            'filename': filename,
                            'img': postimg,
                            'vote_average': 0,  # Default value for sorting
                            'runtime': 0,  # Default value for sorting
                        }
                        results.append(result_entry)
            else:
                print(f"Document ID {doc_item} not found in docID_map.")

        # Calculate vote_average and runtime for each result
        vote_average_column, runtime_column = self.read_additional_info(filename)
        self.calculate_vote_average_and_runtime(results, vote_average_column, runtime_column)

        return results

    def calculate_vote_average_and_runtime(self, results, vote_average_column, runtime_column):
        for result, vote_average, runtime in zip(results, vote_average_column, runtime_column):
            if vote_average and runtime:
                result['vote_average'] = float(vote_average)
                result['runtime'] = int(runtime)

    def read_additional_info(self, filename):
        file_path = os.path.join(self.data_path, filename)
        with open(file_path, encoding='utf-8') as file:
            lines = file.readlines()

            # Assuming the first line contains the headers
            headers = lines[0].strip().split('\t')
            vote_average_index = headers.index('vote_average') if 'vote_average' in headers else -1
            runtime_index = headers.index('runtime') if 'runtime' in headers else -1

            if vote_average_index == -1 or runtime_index == -1:
                print("Error: 'vote_average' or 'runtime' column not found in document.")
                return [], []

            # Extracting the vote_average and runtime columns
            vote_average_column = [line.strip().split('\t')[vote_average_index] if len(line.strip().split('\t')) > vote_average_index else ''
                                   for line in lines[1:]]
            runtime_column = [line.strip().split('\t')[runtime_index] if len(line.strip().split('\t')) > runtime_index else ''
                              for line in lines[1:]]

            return vote_average_column, runtime_column
    def search_documents(self, query):
        query_list = query
        postings = []
        query_output = []

        with open(data_path + "/dict.json", "rb") as iv:
            ivdict = json.load(iv)

        for query in query_list:
            if query in ivdict:
                postings.append([ivdict[query], len(ivdict[query])])
            else:
                print("No results found")
                return []

        postings = sorted(postings, key=itemgetter(1), reverse=False)

        first_list = [posting[0] for posting in postings[0][0]]

        for posting in postings:
            second_list = [list_item[0] for list_item in posting[0]]
            fp = 0
            sp = 0
            while fp < len(first_list) and sp < len(second_list):
                if first_list[fp] == second_list[sp]:
                    if first_list[fp] not in query_output:
                        query_output.append(first_list[fp])
                    fp += 1
                    sp += 1
                elif first_list[fp] < second_list[sp]:
                    fp += 1
                else:
                    sp += 1

        return query_output
    def suggest_movies(self, prefix):
        suggestions = set()
        for filename in self.files:
            titles, _ = self.read_file(filename)
            for title in titles:
                if title.lower().startswith(prefix.lower()):
                    suggestions.add(title)

        return list(suggestions)
    def get_current_movie_info(self, title, filename):
        titles, overviews, ratings, homepages, popularity, taglines, runtimes = self.read_file1(filename)
        self.glfilename = filename
        try:
            index1 = titles.index(title)
        except ValueError:
            return None
        postimg = self.poster(title)
        current_movie_info = {
            'title': titles[index1],
            'overview': overviews[index1],
            'img': postimg,
            'rating': ratings[index1],
            'homepage': homepages[index1],
            'popularity': popularity[index1],
            'tagline': taglines[index1],
            'runtime': runtimes[index1],
        }
        return current_movie_info

    def read_file1(self, filename):
        file_path = os.path.join(self.data_path, filename)
        with open(file_path, encoding='utf-8') as file:
            lines = file.readlines()

            # Assuming the first line contains the headers
            headers = lines[0].strip().split('\t')
            title_index = headers.index('title') if 'title' in headers else -1
            overview_index = headers.index('overview') if 'overview' in headers else -1
            rating_index = headers.index('rating') if 'rating' in headers else -1
            homepage_index = headers.index('homepage') if 'homepage' in headers else -1
            popularity_index = headers.index('popularity') if 'popularity' in headers else -1
            tagline_index = headers.index('tagline') if 'tagline' in headers else -1
            runtime_index = headers.index('runtime') if 'runtime' in headers else -1

            # Extracting the columns
            titles = [line.strip().split('\t')[title_index] if len(line.strip().split('\t')) > title_index else '' for
                      line in lines[1:]]
            overviews = [
                line.strip().split('\t')[overview_index] if len(line.strip().split('\t')) > overview_index else '' for
                line in lines[1:]]
            ratings = [line.strip().split('\t')[rating_index] if len(line.strip().split('\t')) > rating_index else ''
                       for line in lines[1:]]
            homepages = [
                line.strip().split('\t')[homepage_index] if len(line.strip().split('\t')) > homepage_index else '' for
                line in lines[1:]]
            popularity = [
                line.strip().split('\t')[popularity_index] if len(line.strip().split('\t')) > popularity_index else ''
                for line in lines[1:]]
            taglines = [line.strip().split('\t')[tagline_index] if len(line.strip().split('\t')) > tagline_index else ''
                        for line in lines[1:]]
            runtimes = [line.strip().split('\t')[runtime_index] if len(line.strip().split('\t')) > runtime_index else ''
                        for line in lines[1:]]

        return titles, overviews, ratings, homepages, popularity, taglines, runtimes

    def extract_file_summary(self, filename):
        file_path = os.path.join(self.data_path, filename)

        try:
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        except FileNotFoundError:
            return {'error': 'File not found.'}
        except pd.errors.EmptyDataError:
            return {'error': 'Empty file.'}
        except pd.errors.ParserError:
            return {'error': 'Error parsing file.'}

        # Extracting summary information
        summary = {
            'filename': filename,
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'movie_titles': df['title'].tolist() if 'title' in df.columns else [],
        }

        return summary
def extract_movie_genres():
    genres = {}
    added_movies = set()
    file_path = os.path.join(search_app.data_path, search_app.glfilename)
    with open(file_path, encoding='utf-8') as file:
        header = file.readline().strip().split('\t')
        genres_column_index = header.index('genres')
        title_column_index = header.index('title')
        for line in file:
            data = line.strip().split('\t')
            if len(data) > title_column_index and len(data) > genres_column_index:
                title = data[title_column_index]
                if title in added_movies:
                    continue
                movie_genres = eval(data[genres_column_index])
                # time.sleep(0.6)
                # poster_path = 'https://image.tmdb.org/t/p/original/aklNJoFSgcCSQvpG8ssa5OV6rRQ.jpg'
                for genre in movie_genres:
                    genre_name = genre['name']
                    if genre_name not in genres:
                        genres[genre_name] = []
                    if len(genres[genre_name]) < 8:
                        poster_path = search_app.poster(title)
                        genres[genre_name].append({'title': title, 'poster_path': poster_path})

                added_movies.add(title)
        sorted_genres = dict(sorted(genres.items(), key=operator.itemgetter(0)))
        for movies in sorted_genres.values():
            movies.sort(key=lambda x: x['title'])
    return sorted_genres
#     genres = {}
#     file_path = os.path.join(search_app.data_path, search_app.glfilename)
#     with open(file_path, encoding='utf-8') as file:
#         # Assuming your file has a header row
#         header = file.readline().strip().split('\t')
#
#         genres_column_index = header.index('genres')
#         title_column_index = header.index('title')
#
#         for line in file:
#             data = line.strip().split('\t')
#             if len(data) > title_column_index and len(data) > genres_column_index:
#                 title = data[title_column_index]
#                 movie_genres = eval(data[genres_column_index])
#                 poster_path = search_app.poster(title)  # Fetch the poster path
#                 # time.sleep(0.6)
#                 # poster_path = 'https://image.tmdb.org/t/p/original/aklNJoFSgcCSQvpG8ssa5OV6rRQ.jpg'
#                 for genre in movie_genres:
#                     genre_name = genre['name']
#                     if genre_name not in genres:
#                         genres[genre_name] = []
#
#                     genres[genre_name].append({'title': title, 'poster_path': poster_path})
#
#         # Sort genres and movies by genre name
#     sorted_genres = dict(sorted(genres.items(), key=operator.itemgetter(0)))
#     for movies in sorted_genres.values():
#         movies.sort(key=lambda x: x['title'])
#     return sorted_genres
search_app = SearchApp()

@app.route('/')
def index():
    return render_template('index.html', search_results=None)

@app.route('/search', methods=['POST'])
def perform_search():
    start_time = time.time()
    query = request.form.get('search_query')
    if query:
        search_results = search_app.search(query)

        # Get the selected sorting option from the form
        sort_option = request.form.get('sort_option')
        if sort_option == 'ratings':
            search_results = sorted(search_results, key=lambda x: x.get('vote_average', 0), reverse=True)
        elif sort_option == 'runtime':
            search_results = sorted(search_results, key=lambda x: x.get('runtime', 0), reverse=True)

        end_time = time.time()
        time_taken_us = int((end_time - start_time) * 1e6)  # Convert to microseconds
        return render_template('index.html', search_results=search_results, search_time_us=time_taken_us)
    return render_template('index.html', search_results=None, search_time_us=None)
@app.route('/suggest')
def suggest():
    query = request.args.get('q')
    if query:
        suggestions = search_app.suggest_movies(query)
        print("Suggestions:", suggestions)
        return jsonify(suggestions)
    return jsonify([])
@app.route('/info_page')
def info_page():
    title = request.args.get('title')
    filename = request.args.get('filename')
    current_movie_info = search_app.get_current_movie_info(title, filename)
    movie_genres = extract_movie_genres()
    return render_template('info_page.html', current_movie_info=current_movie_info, movie_genres=movie_genres)
    # return render_template('info_page.html', current_movie_info=current_movie_info)


@app.route('/download_summary')
def download_summary():
    filename = search_app.glfilename
    file_path = os.path.join(search_app.data_path, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    df = pd.read_csv(file_path, sep='\t')
    summary = {
        'number_of_rows': len(df),
        'number_of_columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'movie_titles': df['title'].tolist() if 'title' in df.columns else [],
    }
    summary_file_path = os.path.join(search_app.data_path, 'summary.txt')
    with open(summary_file_path, 'w') as summary_file:
        for key, value in summary.items():
            summary_file.write(f"{key}: {value}\n")
    return send_file(summary_file_path, as_attachment=True, download_name='summary.txt')


@app.route('/download_file')
def download_file():
    filename = search_app.glfilename
    file_path = os.path.join(search_app.data_path, filename)
    return send_file(file_path, as_attachment=True)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
