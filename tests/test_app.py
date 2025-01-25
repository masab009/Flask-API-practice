from webapp import app

def testHome():
    assert app.home() == "<h2>roBERTA sentiment analysis</h2>", "home function failed"


