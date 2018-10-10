from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
import time


class TweetHarvester:
    def __init__(self):
        # Fire up the Twitter listener
        access_token = "221621693-eoyVI9TELG1CZ4jJTULolg062x2Q2BKKjS8iH03Y"
        access_token_secret = "xwGD3QUyudlDUgMbEubXNeiAY762WTVsCb37xWOt6t4NR"
        consumer_key = "tUnodnDrtM9k25jKteJejWl5j"
        consumer_secret = "7JLYPk4gg7OxJdmL7umT3jaUns62Aaq8N1OvizvJBOPeQN9f7c"
        self.listener = TweetListener()
        self.latest_tweet = None
        self.tweetlist = []
        self.writeblock = 10  # Batch size of tweets before appending to JSON file
        self.auth = OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_token_secret)
        self.stream = Stream(self.auth, self.listener)
        # Initialize JSON file with an empty list
        with open('tweets.json', 'w', encoding='utf-8') as f:
            json.dump([], f)

    def get_batches(self):
        counter = 0
        while counter < self.writeblock:
            if self.listener.latest_tweet is False:
                print("RATE-LIMITED, STOPPING STREAM")
                self.stream.disconnect()
                return False
            elif self.listener.latest_tweet is not None and self.listener.latest_tweet != self.latest_tweet:
                print(self.listener.latest_tweet)
                self.tweetlist.append(self.listener.latest_tweet)
                self.latest_tweet = self.listener.latest_tweet
                counter += 1
            else:
                pass
        else:
            with open('tweets.json', 'r+') as f:
                data = json.load(f)
            data = data + self.tweetlist
            with open('tweets.json', 'w') as f:
                json.dump(data, f)
            self.tweetlist = []
            return True

    def harvest(self, timer):
        self.stream.filter(locations=[-124.84, 24.39, -66.88, 49.38], async=True)
        time_end = time.time() + timer
        while time.time() < time_end:
            if not self.get_batches():
                break
        else:
            self.stream.disconnect()
            print("ALL FINISHED.")

class TweetListener(StreamListener):
    """
    Open and maintain a persistent connection to the Twitter data stream.
    """
    def __init__(self):
        StreamListener.__init__(self)
        self.latest_tweet = None

    def on_data(self, data):
        """Parse tweets as they arrive."""
        tweet = json.loads(data)
        if 'lang' in tweet and tweet['lang'] == "en":
            text = tweet['extended_tweet']['full_text'] if tweet['truncated'] else tweet['text']
            user = tweet['user']['screen_name']
            coordinates = tweet['coordinates'] if tweet['coordinates'] is not None else "none"
            geo = tweet['geo'] if tweet['geo'] is not None else "none"
            place = tweet['place'] if tweet['place'] is not None else "none"
            tweet_deets = {'text': text, 'user': user, 'place': place, 'geo': geo, 'coordinates': coordinates}
            self.latest_tweet = tweet
        return True


    def on_error(self, status):
        """If the keyword triggers rate-limiting, stop and close listener."""
        if status == 420:
            self.latest_tweet = False
            return False

try:
    tweets = TweetHarvester()
    tweets.harvest(60)
except Exception as e:
    # traceback.print_exc(file=sys.stdout)
    print(e)