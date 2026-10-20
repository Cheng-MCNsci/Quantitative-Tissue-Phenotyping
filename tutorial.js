// Progressive enhancement only: the entire tutorial remains readable without JS.
for(const button of document.querySelectorAll('.copy')){
  button.addEventListener('click',async()=>{
    const code=button.closest('.code-block').querySelector('pre code').textContent;
    try{
      if(navigator.clipboard && window.isSecureContext)await navigator.clipboard.writeText(code);
      else{
        const field=document.createElement('textarea');field.value=code;
        field.style.position='fixed';field.style.opacity='0';document.body.append(field);
        field.select();if(!document.execCommand('copy'))throw Error('copy unavailable');field.remove();
      }
      button.textContent='Copied';setTimeout(()=>button.textContent='Copy',1500);
    }catch{
      button.textContent='Select to copy';
      const range=document.createRange();range.selectNodeContents(button.closest('.code-block').querySelector('pre code'));
      window.getSelection().removeAllRanges();window.getSelection().addRange(range);
    }
  });
}
// Lightweight highlighting without a network dependency.
for(const node of document.querySelectorAll('.code-block pre code')){
  const tokens=node.textContent.split(/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|#[^\n]*|\b(?:from|import|as|for|in|if|else|assert|True|False|None|and|or)\b)/g);
  node.replaceChildren(...tokens.map(value=>{
    const span=document.createElement('span');span.textContent=value;
    if(value.startsWith('#'))span.className='comment';
    else if(value.startsWith('"')||value.startsWith("'"))span.className='str';
    else if(/^(from|import|as|for|in|if|else|assert|True|False|None|and|or)$/.test(value))span.className='kw';
    return span;
  }));
}
const links=[...document.querySelectorAll('.toc li a')];
if('IntersectionObserver' in window){
  const observer=new IntersectionObserver(entries=>{
    for(const entry of entries)if(entry.isIntersecting){
      links.forEach(a=>a.classList.toggle('active',a.hash==='#'+entry.target.id));
    }
  },{rootMargin:'-90px 0px -65% 0px'});
  document.querySelectorAll('main h2').forEach(h=>observer.observe(h));
}
