import praw
import csv
from praw.models import MoreComments


reddit = praw.Reddit(
    user_agent="Comment Extraction (by u/pommes__de__terre)",
    client_id="", #use your reddit api client id
    client_secret="", #use your reddit api client secret key
    username="", #use your username
    password="" #use your password
)
print("We have started")

urls = ["https://www.reddit.com/r/AskReddit/comments/cpzvbu/what_is_your_strongest_held_opinion/"] # put in your urls
		
results = []
with open('test.tsv', 'wt', encoding="utf-8") as tsvfile: #give your output file a name
    writer = csv.writer(tsvfile, delimiter='\t')
    for url in urls:
        submission = reddit.submission(url=url)

        submission.comments.replace_more(limit=5000)
        comment_queue = submission.comments[:]  # Seed with top-level
        while comment_queue:
            comment = comment_queue.pop(0)
            try:
                results.append(comment.body)
                if len(comment.body)>100:
                    writer.writerow([comment.body])
                print(comment.body)
                comment_queue.extend(comment.replies)
            except:
                print("Some error occurred")

#    for i in results:
#        if len(i)>30:
#            writer.writerow([i])