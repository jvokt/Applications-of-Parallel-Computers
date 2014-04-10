.PHONY: run clean realclean tgz

UCI=https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/

run:
	julia driver.jl

anchor_topic.md: anchor_topic.jl
	awk -f doc.awk anchor_topic.jl > anchor_topic.md

anchor_topic.pdf: anchor_topic.md
	pandoc -o $@ $<

clean:
	rm -f anchor_topic.md anchor_topic.pdf

realclean: clean
	rm -f topics*.txt

fetch_kos:
	( cd uci ; wget $(UCI)/docword.kos.txt.gz )
	( cd uci ; wget $(UCI)/vocab.kos.txt )

fetch_nips:
	( cd uci ; wget $(UCI)/docword.nips.txt.gz )
	( cd uci ; wget $(UCI)/vocab.nips.txt )

fetch_enron:
	( cd uci ; wget $(UCI)/docword.enron.txt.gz )
	( cd uci ; wget $(UCI)/vocab.enron.txt )

fetch_nytimes:
	( cd uci ; wget $(UCI)/docword.nytimes.txt.gz )
	( cd uci ; wget $(UCI)/vocab.nytimes.txt )

tgz:
	ln -s . topics
	tar -czf topics.tgz \
		`cat topics/manifest.txt | awk '{ print "topics/" $$1}' ` 
	rm -f topics
