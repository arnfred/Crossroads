#
# Module for searching for papers given a term
#
##########################
import query
import json

def search_papers(search_input) :

    """ Searches the arXiv papers to see if the search input matches authors,
    titles or ids. A list is returned with the results. For example:
        [{ "id" : "dsfgsd.dsfgsdfgs",
           "title" : Blah blah blah,
           "authors" : ["Mr. Blah", "Dr. Blup"]},
         { "id" : "btbtbt.btbtbtbt",
           "title" : Ding ding ding ding,
           "authors" : ["Mr. Ding Dong", "Dr. Swim Swam"]}
        ]
    """


    terms = search_input.split(" ")

    # First check if we have a valid ID in which case only return that result
    if len(terms) == 1 :
        try :
            paper_id = str(terms[0].strip())
            print("PAPER ID %s" % paper_id)
            data = query.get(paper_id)
            print(data)
            return json.dumps([{
                "id" : paper_id,
                "title" : data["title"],
                "authors" : data["authors"]
            }])
        except query.NonExistentID :
            pass # What exception does recommender throw if id doesn't exist?

    # Next check if any terms contains a valid author last name
    # TODO

    # Finally, if a term did match an author, search only papers by this author
    # and return results where the title matches any of the terms. TODO

    # If the term didn't match an author, return papers where the title matches
    # as many terms as possible. TODO

    # For now, return the following result
    temporary_result = json.dumps([
        {
            "id" : "1304.5220",
            "title" : "Scaling Exponent of List Decoders with Applications to Polar Codes",
            "authors" : ["Mr. Mox"]
        },
        {
            "id" : "1305.0547",
            "title" : "On Achievable Rates for Channels with Mismatched Decoding",
            "authors" : ["Dr. Who", "Mr. Mox"]
        },
        {
            "id" : "1301.6120",
            "title" : "A Rate-Splitting Approach to Fading Channels with Imperfect Channel-State Information",
            "authors" : ["Dr. Who", "Mr. Mox"]
        },
        {
            "id" : "1006.2498",
            "title" : "On the Deterministic Code Capacity Region of an Arbitrarily Varying Multiple-Access Channel Under List Decoding",
            "authors" : ["Dr. Who", "Mr. Mox"]
        }]
    )

    return temporary_result






