const hiddenPrediction = () => {
  const resultTag = document.querySelector('.result');
  resultTag.classList.add('hidden');
};

const displayHandlingPrediction = () => {
  const resultTag = document.querySelector('.result');
  resultTag.classList.remove('hidden');
  resultTag.innerText = `預測中...`;
};

const displayPrediction = (result) => {
  const resultTag = document.querySelector('.result');
  resultTag.classList.remove('hidden');
  resultTag.innerText = `預測結果：${result}`;
};

const toggleFeedbackFormHidden = () => {
  const feedbackFormTag = document.querySelector('.feedback');
  feedbackFormTag.classList.toggle('hidden');
};

const getLabels = async () => {
  const response = await fetch('/api/model/boards');
  const data = await response.json();

  return data;
};

const initializeFeedbackOptions = async () => {
  const labels = await getLabels();

  const options = document.querySelector('.options');

  labels
    .map((label) => {
      const li = document.createElement('li');
      const btn = document.createElement('button');

      btn.innerText = label;
      btn.className = 'option';
      btn.type = 'submit';
      btn.name = label;

      li.appendChild(btn);

      return li;
    })
    .forEach((element) => {
      options.appendChild(element);
    });
};

const predictTitleBelongsTo = async (title) => {
  const response = await fetch(`/api/model/prediction?title=${title}`);
  const data = await response.json();
  return data;
};

const sendFeedbackPrompt = async (title, label) => {
  const response = await fetch(
    `/api/model/feedback?title=${title}&label=${label}`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' } }
  );
  const data = await response.json();
  return data;
};

const registerPredictionSubmit = () => {
  const predictionTag = document.querySelector('.prediction');

  predictionTag.addEventListener('submit', async (event) => {
    event.preventDefault();
    const title = document.querySelector('.input').value;

    displayHandlingPrediction();
    const { prediction } = await predictTitleBelongsTo(title);
    displayPrediction(prediction);

    const feedbackPromptTag = document.querySelector('.feedback-prompt');
    feedbackPromptTag.classList.remove('hidden');
  });
};

const registerToggleFeedback = () => {
  const feedbackPromptTag = document.querySelector('.feedback-prompt');

  feedbackPromptTag.addEventListener('click', toggleFeedbackFormHidden);
};

const registerFeedbackSubmit = (label) => {
  const feedbackTag = document.querySelector('.feedback');

  feedbackTag.addEventListener('submit', async (event) => {
    event.preventDefault();
    const title = document.querySelector('.input').value;

    const { message } = await sendFeedbackPrompt(title, event.submitter.name);
    window.alert(message);
  });
};

(async () => {
  await initializeFeedbackOptions();
  registerPredictionSubmit();
  registerFeedbackSubmit();
  registerToggleFeedback();
})();
