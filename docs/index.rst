.. SlickML documentation master file, created by sphinx-quickstart on Sat May 28 00:59:59 2022.
   Reference: 
      - https://sphinx-tutorial.readthedocs.io/
      - https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
      - https://sphinx-design.readthedocs.io/en/furo-theme/
      - https://sphinx-design.readthedocs.io/en/furo-theme/cards.html

.. NOTES: 
   1) You can create a new page under `docs/pages/` in `.md` or `.rst` format
   2) You can add any required sphinx extension to `docs/conf.py`
   3) `docs/pages/releases.md` is a symlink to `CHANGELOG.md` in the root

.. TODOS:
   1) Once we add example notebooks, we can populate them below. I am not still sure on the format
   maybe we can have one examples pages and add the examples there as table with a permalink to
   github project repo. Anything we put in the doc, should be easy to maintain. So, we should
   maximize the use of permalinks and symlinks.

SlickML🧞 Documentation
*******************************

|build_status| |docs_status| |codecov| |downloads| |github_stars| |slack_invite| |twitter_url|

----

🧠 SlickML🧞 Philosophy
-----------------------------

`SlickML <https://github.com/slickml/slick-ml>`_ is an open-source machine learning library written in Python aimed at accelerating the experimentation time for ML applications
with tabular data while maximizing the amount of information can be inferred. Data Scientists' tasks can often be repetitive such as feature
selection, model tuning, or evaluating metrics for classification and regression problems. We strongly believe that a good portion of the tasks
based on tabular data can be addressed via gradient boosting and generalized linear models `[1] <https://arxiv.org/pdf/2207.08815.pdf>`_.
SlickML provides Data Scientists with a toolbox to quickly prototype solutions for a given problem with minimal code while maximizing the amound
of information that can be inferred. Additionally, the prototype solutions can be easily promoted and served in production with our recommended
recipes via various model serving frameworks including `ZenML <https://github.com/zenml-io/zenml>`_, `BentoML <https://github.com/bentoml/BentoML>`_,
and `Prefect <https://github.com/PrefectHQ/prefect>`_. More details coming soon 🤞 ...


.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`🛠 Installation <pages/installation>`
        :link: pages/installation
        :link-type: doc

        Learn more about the requirements and how to set up your Python environment to install SlickML smoothly ...

    .. grid-item-card:: :doc:`📌 Quick Start <pages/quick_start>`
        :link: pages/quick_start
        :link-type: doc

        Wanna take a glimpse at some of the key functionalities? Quick start is the best place to start your journey ...

    .. grid-item-card:: :doc:`🎯 API Reference <pages/api>`
        :link: pages/api
        :link-type: doc

        Explore SlickML API Reference and hopefully scrutinize the source code ...

    .. grid-item-card:: :doc:`📣 Changelog & Releases <pages/releases>`
        :link: pages/releases
        :link-type: doc

        Stay up-to-date with the new features in SlickML by checking out the latest release notes ...      
----

🧑‍💻🤝 Become a Contributor
----------------------------
SlickML🧞 is trying to build a thriving open source community where data scientists and machine learning
practitioners can come together and contribute their ideas to the project. The details of the development
process in are laid out in our `Contributing <pages/contributing.html>`_ guidelines. We strongly believe that
reading and following these guidelines will help us make the contribution process easy and effective for
everyone involved 🚀🌙 . Special thanks to all of our amazing contributors 👇

.. image:: https://contrib.rocks/image?repo=slickml/slick-ml
  :width: 300
  :alt: Contributors
  :target: https://github.com/slickml/slick-ml/graphs/contributors

.. image:: https://repobeats.axiom.co/api/embed/ca865991b0547199fe7a069de7af25645b225e9c.svg
  :width: 1000
  :alt: Repobeats analytics image
  :target: https://github.com/slickml/slick-ml/commits/master

----

❓ 🆘 📲 Need Help?
----------------------
Please join our `Slack Channel <https://www.slickml.com/slack-invite>`_ to interact directly with the core team and our small
community. This is a good place to discuss your questions and ideas or in general ask for help 👨‍👩‍👧 👫 👨‍👩‍👦 .




.. toctree::
   :hidden:
   :maxdepth: 1
   
   Installation <pages/installation>
   Quick Start <pages/quick_start>
   Releases <pages/releases>
   Contributing <pages/contributing>
   Citation <pages/citation>
   License <pages/license>
   Code of Conduct <pages/code_of_conduct>
   Contact Us <pages/contact_us>
   API Reference <pages/api>

.. FUTURE PAGES:
   Examples <pages/examples>

----

🔍 Indices and Tables
-----------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. |build_status| image:: https://github.com/slickml/slick-ml/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/slickml/slick-ml/actions/workflows/ci.yml
.. |docs_status| image:: https://github.com/slickml/slick-ml/actions/workflows/cd.yml/badge.svg
   :target: https://github.com/slickml/slick-ml/actions/workflows/cd.yml
.. |codecov| image:: https://codecov.io/gh/slickml/slick-ml/graph/badge.svg?token=Z7XP51MB4K
   :target: https://codecov.io/gh/slickml/slick-ml
.. |license| image:: https://img.shields.io/github/license/slickml/slick-ml
   :target: https://github.com/slickml/slick-ml/blob/master/LICENSE
.. |downloads| image:: https://pepy.tech/badge/slickml
   :target: https://pepy.tech/project/slickml
.. |twitter_url| image:: https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2FSlickML
   :target: https://twitter.com/SlickML
.. |slack_invite| image:: https://badgen.net/badge/Join/SlickML%20Slack/purple?icon=slack
   :target: https://www.slickml.com/slack-invite
.. |github_stars| image:: https://img.shields.io/github/stars/slickml/slick-ml?color=cyan&label=github&logo=github
   :target: https://github.com/slickml/slick-ml
