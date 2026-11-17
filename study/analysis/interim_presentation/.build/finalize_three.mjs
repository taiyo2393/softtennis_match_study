import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const build=path.dirname(fileURLToPath(import.meta.url)),workspaceDir=path.dirname(build);
const skill='C:/Users/ta239/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const python='C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';process.env.RUNTIME_NODE=process.execPath;process.env.RUNTIME_PYTHON=python;
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const finalPath=path.join(workspaceDir,'output','23238007R_アニメーション3-5ページ_修正版.pptx');
await finalizePresentation({workspaceDir,candidatePath:path.join(build,'three_candidate.pptx'),finalPath,explicitTotalSlideCount:3,pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','9144000,6858000','--validate-heading-fit'],verifyArtifactToolImport:true,receiptPath:path.join(build,'three_validation.json')});
const deck=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.mkdir(path.join(build,'three_render'),{recursive:true});
for(let i=0;i<deck.slides.items.length;i++){const b=await deck.export({slide:deck.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(build,'three_render',`${i+1}.png`),new Uint8Array(await b.arrayBuffer()));}
console.log(finalPath);
