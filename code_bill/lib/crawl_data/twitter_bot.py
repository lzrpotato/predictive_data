from copy import Error
import os
import re
import time
import string
import sys
sys.path.append('~/Software/geckodriver')

from selenium import webdriver
from selenium.webdriver import ActionChains, FirefoxOptions, ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
#from selenium.webdriver.firefox.webdriver import WebDriver
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

'''Uncomment the below line when running in linux'''
#from pyvirtualdisplay import Display 


def get_credentials(path) -> dict: 
    # dictionary for storing credentials 
    credentials = dict() 
    # reading the text file  
    # for credentials 
    with open(path,'r') as f: 
        # interating over the lines 
        for line in f.readlines(): 
            try: 
                # fetching email and password 
                key, value = line.split(": ") 
            except ValueError: 
                # raises error when email and password not supplied 
                print('Add your email and password in credentials file') 
                exit(0) 
            # removing trailing  
            # white space and new line 
            credentials[key] = value.rstrip(" \n") 
    # returning the dictionary containing the credentials 
    return credentials 

def replace_url(test_str):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    return re.sub(regex,"URL", test_str)

def count_world_in_str(test_str):
    res = re.findall(r'\w+', test_str)
    count = len(res)
    return count

class Twitterbot: 
    def __init__(self, email=None, password=None, driver_type='firefox'): 

        """Constructor 

        Arguments: 
            email {string} -- registered twitter email 
            password {string} -- password for the twitter account 
        """

        self.email = email 
        self.password = password 
        # adding the path to the chrome driver and 
        # integrating chrome_options with the bot 
        if driver_type == 'firefox':
            self.bot = self.get_driver()
        elif driver_type == 'chrome':
            self.bot = self.get_chrome_driver()
        else:
            raise ValueError("setup driver")

    def login(self): 
        """ 
            Method for signing in the user 
            with the provided email and password. 
        """
        bot = self.bot
        # fetches the login page 
        bot.get('https://twitter.com/login')
        # adjust the sleep time according to your internet speed 
        email = WebDriverWait(bot,4).until(EC.presence_of_element_located((By.NAME,'session[username_or_email]')))
        password = bot.find_element_by_name('session[password]')

        # sends the email to the email input 
        email.send_keys(self.email) 
        # sends the password to the password input 
        password.send_keys(self.password) 
        # executes RETURN key action 
        password.send_keys(Keys.RETURN)
        #action = ActionChains(bot)
        #submit = bot.find_element_by_xpath('//div[@data-testid="LoginForm_Login_Button"]')
        #action.move_to_element(submit).perform()
        #submit.click()
        time.sleep(2)
        bot.save_screenshot('login.png')
        print('login finished')

    def close(self):
        self.bot.close()

    def get_chrome_driver(self):
        chrome_options = ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        if not os.path.isdir('./lib/crawl_data/logging/'):
            os.mkdir('./lib/crawl_data/logging/')
        driver = webdriver.Chrome('./lib/crawl_data/chromedriver',chrome_options=chrome_options,service_log_path='./lib/crawl_data/logging/')

        return driver


    def get_driver(self):
        options = FirefoxOptions()
        options.add_argument("--headless")
        options.add_argument("--load-images=no")

        
        if not os.path.isdir('./firefoxprofiles/'):
           os.mkdir('./firefoxprofiles/')
        if not os.path.isdir('./lib/crawl_data/firefoxprofiles/'):
            os.mkdir('./lib/crawl_data/firefoxprofiles/')
        firefox_profile = webdriver.FirefoxProfile('./lib/crawl_data/firefoxprofiles/')
        profile_dict = {'permissions.default.image': 2}
        for k,v in profile_dict.items():
            firefox_profile.set_preference(k,v)
        
        firefox_binary = FirefoxBinary('/home/zluo_epscor/Software/firefox/firefox')
        driver = webdriver.Firefox(
            executable_path='/home/zluo_epscor/Software/geckodriver',
            firefox_binary=firefox_binary,
            firefox_profile=firefox_profile,
            options=options
            )
        return driver

    def find_tweet_by_id(self, id):
        base_url = "https://twitter.com/anyuser/status/" + str(id)
        bot = self.bot
        bot.get(base_url)
        start_time=time.time()
        #print('this is start_time ',start_time)
        #driver.find_element_by_id("kw").send_keys("selenium webdriver")
        #driver.find_element_by_id("su").click()
        
        # get content of the tweet
        content = WebDriverWait(bot,4).until(EC.presence_of_element_located((By.XPATH,'//div[@lang="en"]'))).text
        #bot.save_screenshot('screen.png')
        #print(content)
        
        # get the user who wrote the tweet
        user = bot.find_element_by_xpath('//div[@dir="ltr"]')
        user_name = user.text.split('@')[1]
        #print(user_name)

        # get the tweeting time of the tweet
        tweet_time = bot.find_element_by_xpath(f'//div/a[@href="/{user_name}/status/{id}"]').text
        #print(tweet_time)

        end_time=time.time()
        #print('this is end_time ',end_time)
        #print('run time ', (end_time-start_time))
        #bot.close()
        return replace_url(content), user_name, tweet_time

    def find_profile_by_name(self, username):
        url = 'https://twitter.com/' + str(username)
        bot = self.bot
        bot.get(url)
        return self.extract_profile(bot)

    def find_profile_by_id(self, id):
        url = 'https://twitter.com/i/user/' + str(id)
        bot = self.bot
        bot.get(url)
        return self.extract_profile(bot)

    def extract_profile(self, bot: WebDriver):
        profile = {}
        start_time=time.time()
        #print('this is start_time ',start_time)

        # number of words in a user's self-description,
        description = WebDriverWait(bot,4).until(EC.presence_of_element_located(
            (By.XPATH,'//div[@data-testid="UserDescription"]')
            )).text
        #print(repr(replace_url(description.replace('\n',''))))
        profile['n_description'] = count_world_in_str(replace_url(description))
        # name
        user = bot.find_element_by_xpath('//div[@dir="ltr"]')
        username = user.text.split('@')[1]
        # number of words in user's screen name,
        screen_name = bot.find_element_by_xpath('//div/div/div/span/span').text
        #print(screen_name)
        profile['n_screen_name'] = count_world_in_str(screen_name)
        # number of users who follows user,
        follower = bot.find_element_by_xpath(f'//a[@href="/{username}/followers"]/span/span').text
        #print(follower)
        profile['follower'] = follower
        # number of users that uj is following,
        following = bot.find_element_by_xpath(f'//a[@href="/{username}/following"]/span/span').text
        #print(following)
        profile['following'] = following
        # number of created stories for user
        created_story = bot.find_element_by_xpath('//div[@class="css-1dbjc4n r-1habvwh"]/div[@dir="auto"]').text
        #print(created_story)
        profile['created_story'] = created_story.split(' ')[0]
        # whether the uj account is verified or not
        verified_flag = False
        try:
            verified = bot.find_element_by_xpath('//div[@aria-label="Provides details about verified accounts."]')
            #print('verified')
            verified_flag = True
        except:
            #print('not verified')
            verified_flag = False
        profile['verified'] = verified_flag
        # whether uj allows the geo-spatial positioning &&& joined time
        located = False
        located2joined = bot.find_elements_by_xpath('//div[@data-testid="UserProfileHeader_Items"]/span')
        if len(located2joined) == 2:
            #print('located')
            located = True
            joined_t = located2joined[1].text
        else:
            #print('not located')
            located = False
            joined_t = located2joined[0].text
        profile['verified'] = located
        profile['joined'] = joined_t.split(' ')[1:]
        # time difference between the source tweet’s post time and uj’s retweet time
        # the length of retweet path between uj and the source tweet (1 if uj retweets the source tweet)
        end_time=time.time()
        #print('this is end_time ',end_time)
        #print('run time ', (end_time-start_time))
        return profile

def main():
    credentials = get_credentials('credentials.txt')
    bot = Twitterbot(credentials['email'],credentials['password'])
    #bot.login()
    
    content, username, tweet_time = bot.find_tweet_by_id('1339363006854037504')
    profile = bot.find_profile_by_name(username)
    print(content)
    print(profile)
    content, username, tweet_time = bot.find_tweet_by_id('624298742162845696')
    profile = bot.find_profile_by_name(username)
    print(content)
    print(profile)
    
    bot.close()

if __name__ == '__main__':
    main()
