# -- Driver for the anchor words

include("anchor_topic.jl")
require("anchor_topic.jl")

# Load the data
words = readcsv("yelp/words.csv")
Q = readcsv("yelp/Q.csv")

# Run the topic mining
(p, r, A, TW) = mine_topics(Q)

# Write the results files
write_topics("topics.txt", words, p, r, A, TW)
